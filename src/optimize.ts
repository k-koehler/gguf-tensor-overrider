import {
  GGMLFileQuantizationType,
  GGMLQuantizationType,
  type GGUFParseOutput,
} from "@huggingface/gguf";
import type { Gpu } from "./nvidia.ts";
import Log from "./log.ts";

const GGUFQuantizationSizeMapBytes: Record<GGMLQuantizationType, number> = {
  // Fixed-size types (no block overhead)
  [GGMLQuantizationType.F32]: 4, // 32 bits → 4 bytes
  [GGMLQuantizationType.F16]: 2, // 16 bits → 2 bytes
  [GGMLQuantizationType.BF16]: 2, // 16 bits → 2 bytes
  [GGMLQuantizationType.F64]: 8, // 64 bits → 8 bytes

  // Block-based types (32 elements per block unless specified)
  [GGMLQuantizationType.Q4_0]: 0.5625, // (2 bytes delta + 16 bytes quants) / 32 elements [1]
  [GGMLQuantizationType.Q4_1]: 0.5938, // (2 bytes delta + 19 bytes quants) / 32 elements [1]
  [GGMLQuantizationType.Q5_0]: 0.6875, // (2 bytes delta + 22 bytes quants) / 32 elements [1]
  [GGMLQuantizationType.Q5_1]: 0.7188, // (2 bytes delta + 23 bytes quants) / 32 elements [1]
  [GGMLQuantizationType.Q8_0]: 1.0625, // (2 bytes delta + 32 bytes quants) / 32 elements [1]
  [GGMLQuantizationType.Q8_1]: 1.0938, // (2 bytes delta + 35 bytes quants) / 32 elements [1]

  // Approximations for K-series (block sizes vary; 256 elements assumed)
  [GGMLQuantizationType.Q2_K]: 0.5625, // Approximate as Q4_0 (0.5625 bytes/element)
  [GGMLQuantizationType.Q3_K]: 0.625, // Approximate as midway between Q4_0 and Q5_0 (0.625 bytes/element)
  [GGMLQuantizationType.Q4_K]: 0.625, // Midway between Q4_0 (0.5625) and Q5_0 (0.6875) = 0.625 bytes/element
  [GGMLQuantizationType.Q5_K]: 0.6875, // Approximate as Q5_0 (0.6875 bytes/element)
  [GGMLQuantizationType.Q6_K]: 0.75, // Approximate as slightly above Q5_0 (0.75 bytes/element)

  // Integer types (no quantization overhead)
  [GGMLQuantizationType.I8]: 0.125, // 8 bits → 1 byte per element
  [GGMLQuantizationType.I16]: 0.25, // 16 bits → 2 bytes per element
  [GGMLQuantizationType.I32]: 0.5, // 32 bits → 4 bytes per element
  [GGMLQuantizationType.I64]: 1, // 64 bits → 8 bytes per element

  // IQ variants (approximated using bits per element, no block overhead)
  [GGMLQuantizationType.IQ2_XXS]: 0.25, // 2 bits → 0.25 bytes
  [GGMLQuantizationType.IQ2_XS]: 0.25, // 2 bits → 0.25 bytes
  [GGMLQuantizationType.IQ3_XXS]: 0.1667, // 1.33 bits → ~0.1667 bytes
  [GGMLQuantizationType.IQ1_S]: 0.5, // 4 bits → 0.5 bytes
  [GGMLQuantizationType.IQ4_NL]: 0.125, // 4 bits → 0.5 bytes (original may be incorrect) [1]
  [GGMLQuantizationType.IQ3_S]: 0.1667, // 1.33 bits → ~0.1667 bytes
  [GGMLQuantizationType.IQ2_S]: 0.25, // 2 bits → 0.25 bytes
  [GGMLQuantizationType.IQ4_XS]: 0.125, // 4 bits → 0.5 bytes (original may be incorrect) [1]
  [GGMLQuantizationType.IQ1_M]: 0.5, // 4 bits → 0.5 bytes
  [GGMLQuantizationType.TQ1_0]: 0.5, // 4 bits → 0.5 bytes (assumed)
  [GGMLQuantizationType.TQ2_0]: 0.25,
  [GGMLQuantizationType.Q8_K]: 1, // 8 bits → 1 byte per element (assumed)
};

function bytesToMiB(bytes: number): number {
  return bytes / (1024 * 1024);
}

function extractMetadata(gguf: GGUFParseOutput): {
  hiddenSize: number;
  numAttentionHeads: number;
  numLayers: number;
  numKeyValueHeads: number;
  headSize: number;
} {
  switch (gguf.metadata["general.architecture"]) {
    case "qwen3moe" as string:
      return {
        hiddenSize: gguf.metadata["qwen3moe.embedding_length"],
        numAttentionHeads: gguf.metadata["qwen3moe.attention.head_count"],
        numLayers: gguf.metadata["qwen3moe.block_count"],
        numKeyValueHeads: gguf.metadata["qwen3moe.attention.head_count_kv"],
        headSize:
          gguf.metadata["qwen3moe.embedding_length"] /
          gguf.metadata["qwen3moe.attention.head_count"],
      };
    case "qwen3" as string:
      return {
        hiddenSize: gguf.metadata["qwen3.embedding_length"],
        numAttentionHeads: gguf.metadata["qwen3.attention.head_count"],
        numLayers: gguf.metadata["qwen3.block_count"],
        numKeyValueHeads: gguf.metadata["qwen3.attention.head_count_kv"],
        headSize:
          gguf.metadata["qwen3.embedding_length"] /
          gguf.metadata["qwen3.attention.head_count"],
      };
    default:
      throw new Error(
        `Unsupported architecture: ${gguf.metadata["general.architecture"]}`
      );
  }
}

function calculateKvCacheSizeBytes(
  gguf: GGUFParseOutput,
  contextLength: number,
  contextQuantizationSize: number
): number {
  const { numLayers, numKeyValueHeads, headSize } = extractMetadata(gguf);
  const contextQuantizationByteSize = contextQuantizationSize / 8;
  return (
    2 * // 2 for key and value
    contextQuantizationByteSize * // Size of each element in bytes
    numLayers * // Number of layers
    contextLength * // Context length
    numKeyValueHeads * // Number of key-value heads
    headSize // Head size
  );
}

function calculateTensorSizeBytes(
  tensor: GGUFParseOutput["tensorInfos"][number]
): number {
  const quantizationType = tensor.dtype;
  const quantizationSize = GGUFQuantizationSizeMapBytes[quantizationType];
  if (quantizationSize === undefined) {
    throw new Error(
      `Unsupported quantization type: ${quantizationType} in tensor ${tensor.name}`
    );
  }
  const tensorShapeAsNumber = tensor.shape.map(Number);
  let tensorSize = tensorShapeAsNumber[0];
  for (let i = 1; i < tensorShapeAsNumber.length; i++) {
    tensorSize *= tensorShapeAsNumber[i];
  }
  tensorSize *= quantizationSize;
  return tensorSize;
}

function calculateTensorsSizeBytes(gguf: GGUFParseOutput): number {
  let totalSize = 0;
  for (const tensor of gguf.tensorInfos) {
    totalSize += calculateTensorSizeBytes(tensor);
  }
  return totalSize;
}

function modelFitsInMemory({
  gguf,
  gpus,
  ramBytes,
  contextLength,
  contextQuantizationSize,
}: {
  gguf: GGUFParseOutput;
  gpus: Gpu[];
  ramBytes: number;
  contextLength: number;
  contextQuantizationSize: number;
}): boolean {
  const kvSize = calculateKvCacheSizeBytes(
    gguf,
    contextLength,
    contextQuantizationSize
  );
  const tensorSize = calculateTensorsSizeBytes(gguf);
  const totalModelSize = kvSize + tensorSize;
  const totalGpuMemory = gpus.reduce(
    (acc, gpu) => acc + gpu.memoryTotalBytes,
    0
  );
  const totalRamMemory = ramBytes;
  const totalMemory = totalGpuMemory + totalRamMemory;
  return totalModelSize <= totalMemory;
}

class Device {
  public readonly name: string;
  public readonly memoryTotalBytes: number;
  public readonly priority: number;
  public bytesAllocated: number = 0;
  private unsafe: boolean = false;

  constructor(name: string, memoryTotalBytes: number, priority: number) {
    this.name = name;
    this.memoryTotalBytes = memoryTotalBytes;
    this.priority = priority;
  }

  private get safeMemoryTotalBytes(): number {
    return this.memoryTotalBytes * 0.95;
  }

  public canAllocate(requiredMemoryBytes: number): boolean {
    if (this.unsafe) {
      return true;
    }
    return (
      this.bytesAllocated + requiredMemoryBytes <= this.safeMemoryTotalBytes
    );
  }

  public setUnsafe(): void {
    this.unsafe = true;
  }

  public alloc(requiredMemoryBytes: number): void {
    if (!this.canAllocate(requiredMemoryBytes)) {
      throw new Error(
        `Cannot allocate ${bytesToMiB(requiredMemoryBytes).toFixed(
          2
        )} MiB on device ${this.name}.`
      );
    }
    this.bytesAllocated += requiredMemoryBytes;
  }
}

class DeviceAllocator {
  public devices: Device[];
  public tensorMap: Record<string, string> = {};
  constructor(devices: Device[]) {
    this.devices = devices;
  }

  public allocate(requiredMemoryBytes: number, tensorName?: string): Device {
    const sortedDevices = [...this.devices].sort(
      (a, b) => b.priority - a.priority
    );
    for (const device of sortedDevices) {
      if (device.canAllocate(requiredMemoryBytes)) {
        device.alloc(requiredMemoryBytes);
        if (tensorName) {
          this.tensorMap[tensorName] = device.name;
        }
        return device;
      }
    }
    throw new Error(
      `Cannot allocate ${bytesToMiB(requiredMemoryBytes).toFixed(
        2
      )} MiB on any device.`
    );
  }
}

function tensorsBlockwise(
  gguf: GGUFParseOutput
): GGUFParseOutput["tensorInfos"][] {
  const blocks = [];
  for (const tensor of gguf.tensorInfos) {
    // each tensor should be in format blk.[i].<...>
    // where i is the block index
    const blockName = tensor.name.split(".")[1];
    if (!blockName || isNaN(+blockName)) {
      // continue, there are some special tensors that do not follow the block naming convention
    }
    blocks[blockName] ??= [];
    blocks[blockName].push(tensor);
  }
  return blocks;
}

export default function optimize({
  gguf,
  gpus,
  ramBytes,
  contextLength,
  contextQuantizationSize,
  check = true,
}: {
  gguf: GGUFParseOutput;
  gpus: Gpu[];
  ramBytes: number;
  contextLength: number;
  contextQuantizationSize: number;
  check: boolean;
}): void {
  if (
    check &&
    !modelFitsInMemory({
      gguf,
      gpus,
      ramBytes,
      contextLength,
      contextQuantizationSize,
    })
  ) {
    throw new Error(
      "Model does not fit in combined GPU and RAM memory. Try reducing context length or quantization size."
    );
  }
  const metadata = extractMetadata(gguf);
  const cpuDevice = new Device("CPU", ramBytes, 0);
  if (!check) {
    cpuDevice.setUnsafe();
  }
  const gpuDevices = gpus.map(
    (gpu) => new Device(gpu.name, gpu.memoryTotalBytes, gpu.memoryTotalBytes) // Use GPU memory as priority as a heuristic for computation power
  );
  const allocator = new DeviceAllocator([cpuDevice, ...gpuDevices]);
  const seen = new Set<string>();

  // first pass
  // blockwise enumerate over tensors for the purposes of allocating attention
  // later on we can allocate tensorwise, but kv cache is blockwise
  const attentionTensorNameFlags = ["attention", "attn"];
  const kvCachePerBlock =
    calculateKvCacheSizeBytes(gguf, contextLength, contextQuantizationSize) /
    metadata.numLayers;
  let totalAttentionBytes = 0;
  for (const block of tensorsBlockwise(gguf)) {
    allocator.allocate(kvCachePerBlock);
    totalAttentionBytes += kvCachePerBlock;
    for (const tensor of block) {
      if (
        !attentionTensorNameFlags.some((keyword) =>
          tensor.name.toLowerCase().includes(keyword)
        )
      ) {
        continue;
      }
      const tensorSize = calculateTensorSizeBytes(tensor);
      allocator.allocate(tensorSize, tensor.name);
      seen.add(tensor.name);
      totalAttentionBytes += tensorSize;
    }
  }
  Log.log(
    "info",
    `Total attention bytes allocated: ${bytesToMiB(totalAttentionBytes).toFixed(
      2
    )} MiB`
  );
  Log.log("info", "Device allocation after attention pass:");
  for (const device of allocator.devices) {
    Log.log(
      "info",
      `Device ${device.name}: ${bytesToMiB(device.bytesAllocated).toFixed(
        2
      )} MiB allocated`
    );
  }

  // second pass
  // allocated the ffn tensors minus expert tensors
  const ffnTensorNameFlags = ["ffn", "feed_forward"];
  const ffnTensorNameNoFlags = ["exp", "expert", "gate", "norm"];
  let totalFfnBytes = 0;
  for (const block of tensorsBlockwise(gguf)) {
    for (const tensor of block) {
      if (
        !ffnTensorNameFlags.some((keyword) =>
          tensor.name.toLowerCase().includes(keyword)
        ) ||
        ffnTensorNameNoFlags.some((keyword) =>
          tensor.name.toLowerCase().includes(keyword)
        )
      ) {
        continue;
      }
      if (seen.has(tensor.name)) {
        continue;
      }
      const tensorSize = calculateTensorSizeBytes(tensor);
      allocator.allocate(tensorSize, tensor.name);
      totalFfnBytes += tensorSize;
      seen.add(tensor.name);
    }
  }
  Log.log(
    "info",
    `Total FFN bytes allocated: ${bytesToMiB(totalFfnBytes).toFixed(2)} MiB`
  );
  Log.log("info", "Device allocation after FFN pass:");
  for (const device of allocator.devices) {
    Log.log(
      "info",
      `Device ${device.name}: ${bytesToMiB(device.bytesAllocated).toFixed(
        2
      )} MiB allocated`
    );
  }

  // third pass
  // only relevent for MoE models
  // allocate the expert gate tensors
  const gateTensorNameFlags = ["gate"];
  let totalGateBytes = 0;
  for (const block of tensorsBlockwise(gguf)) {
    for (const tensor of block) {
      if (
        !gateTensorNameFlags.some((keyword) =>
          tensor.name.toLowerCase().includes(keyword)
        )
      ) {
        continue;
      }
      if (seen.has(tensor.name)) {
        continue;
      }
      const tensorSize = calculateTensorSizeBytes(tensor);
      allocator.allocate(tensorSize, tensor.name);
      totalGateBytes += tensorSize;
      seen.add(tensor.name);
    }
  }
  Log.log(
    "info",
    `Total gate bytes allocated: ${bytesToMiB(totalGateBytes).toFixed(2)} MiB`
  );
  Log.log("info", "Device allocation after gate pass:");
  for (const device of allocator.devices) {
    Log.log(
      "info",
      `Device ${device.name}: ${bytesToMiB(device.bytesAllocated).toFixed(
        2
      )} MiB allocated`
    );
  }

  // fourth pass, allocate the norm tensors
  const normTensorNameFlags = ["norm"];
  let totalNormBytes = 0;
  for (const block of tensorsBlockwise(gguf)) {
    for (const tensor of block) {
      if (
        !normTensorNameFlags.some((keyword) =>
          tensor.name.toLowerCase().includes(keyword)
        )
      ) {
        continue;
      }
      if (seen.has(tensor.name)) {
        continue;
      }
      const tensorSize = calculateTensorSizeBytes(tensor);
      allocator.allocate(tensorSize, tensor.name);
      totalNormBytes += tensorSize;
      seen.add(tensor.name);
    }
  }
  Log.log(
    "info",
    `Total norm bytes allocated: ${bytesToMiB(totalNormBytes).toFixed(2)} MiB`
  );
  Log.log("info", "Device allocation after norm pass:");
  for (const device of allocator.devices) {
    Log.log(
      "info",
      `Device ${device.name}: ${bytesToMiB(device.bytesAllocated).toFixed(
        2
      )} MiB allocated`
    );
  }

  // fifth pass, allocate the rest of the tensors
  let totalRestBytes = 0;
  for (const block of tensorsBlockwise(gguf)) {
    for (const tensor of block) {
      if (seen.has(tensor.name)) {
        continue;
      }
      const tensorSize = calculateTensorSizeBytes(tensor);
      allocator.allocate(tensorSize, tensor.name);
      totalRestBytes += tensorSize;
      seen.add(tensor.name);
    }
  }
  Log.log(
    "info",
    `Total rest bytes allocated: ${bytesToMiB(totalRestBytes).toFixed(2)} MiB`
  );
  Log.log("info", "Final device allocation:");
  for (const device of allocator.devices) {
    Log.log(
      "info",
      `Device ${device.name}: ${bytesToMiB(device.bytesAllocated).toFixed(
        2
      )} MiB allocated`
    );
  }

  Log.log("info", "Tensor allocation map:");
  for (const [tensorName, deviceName] of Object.entries(allocator.tensorMap)) {
    Log.log("info", `Tensor ${tensorName} allocated on device ${deviceName}`);
  }

  let command = "-ngl 0 ";
  for (const [tensorName, deviceName] of Object.entries(allocator.tensorMap)) {
    command += `-ot "${tensorName}=${deviceName}" `;
  }
  Log.log("default", command.trim());
}
