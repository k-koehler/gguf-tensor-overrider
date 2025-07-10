import { GGMLQuantizationType, type GGUFParseOutput } from "@huggingface/gguf";
import type { Gpu } from "./nvidia.ts";
import Log from "./log.ts";

const GGUFQuantizationSizeMapBytes: Record<GGMLQuantizationType, number> = {
  // --- Fixed-size types ---
  [GGMLQuantizationType.F32]: 4,
  [GGMLQuantizationType.F16]: 2,
  [GGMLQuantizationType.BF16]: 2,
  [GGMLQuantizationType.F64]: 8,

  // --- Integer types ---
  [GGMLQuantizationType.I8]: 1,
  [GGMLQuantizationType.I16]: 2,
  [GGMLQuantizationType.I32]: 4,
  [GGMLQuantizationType.I64]: 8,

  // --- Block-based Q-types (32 elements per block) ---
  [GGMLQuantizationType.Q4_0]: 0.5625, // (2 + 16) / 32
  [GGMLQuantizationType.Q4_1]: 0.625, // (2 + 2 + 16) / 32
  [GGMLQuantizationType.Q5_0]: 0.6875, // (2 + 20) / 32
  [GGMLQuantizationType.Q5_1]: 0.75, // (2 + 2 + 20) / 32
  [GGMLQuantizationType.Q8_0]: 1.0625, // (2 + 32) / 32
  [GGMLQuantizationType.Q8_1]: 1.125, // (2 + 2 + 32) / 32

  // --- K-series (K-quants) ---
  [GGMLQuantizationType.Q2_K]: 0.359375, // 2.875 bpw
  [GGMLQuantizationType.Q3_K]: 0.4375, // 3.5 bpw
  [GGMLQuantizationType.Q4_K]: 0.5625, // 4.5 bpw
  [GGMLQuantizationType.Q5_K]: 0.6875, // 5.5 bpw
  [GGMLQuantizationType.Q6_K]: 0.8125, // 6.5 bpw
  [GGMLQuantizationType.Q8_K]: 1.0, // 8 bpw

  // --- IQ (Importance-aware Quantization) types ---
  [GGMLQuantizationType.IQ1_M]: 0.1953125, // 1.5625 bpw
  [GGMLQuantizationType.IQ1_S]: 0.22265625, // 1.78125 bpw
  [GGMLQuantizationType.IQ2_XXS]: 0.2734375, // 2.1875 bpw
  [GGMLQuantizationType.IQ2_XS]: 0.3046875, // 2.4375 bpw
  [GGMLQuantizationType.IQ2_S]: 0.3125, // 2.5 bpw
  [GGMLQuantizationType.IQ3_XXS]: 0.40234375, // 3.21875 bpw
  [GGMLQuantizationType.IQ3_S]: 0.44140625, // 3.53125 bpw
  [GGMLQuantizationType.IQ4_NL]: 0.5, // 4.0 bpw
  [GGMLQuantizationType.IQ4_XS]: 0.53125, // 4.25 bpw

  // --- Tensor Quantization (assumed from user context) ---
  [GGMLQuantizationType.TQ1_0]: 0.5, // Assumed 4-bit
  [GGMLQuantizationType.TQ2_0]: 0.25, // Assumed 2-bit
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
    case "llama4" as string:
      return {
        hiddenSize: gguf.metadata["llama4.embedding_length"],
        numAttentionHeads: gguf.metadata["llama4.attention.head_count"],
        numLayers: gguf.metadata["llama4.block_count"],
        numKeyValueHeads: gguf.metadata["llama4.attention.head_count_kv"],
        headSize:
          gguf.metadata["llama4.embedding_length"] /
          gguf.metadata["llama4.attention.head_count"],
      };
    case "llama":
      return {
        hiddenSize: gguf.metadata["llama.embedding_length"],
        numAttentionHeads: gguf.metadata["llama.attention.head_count"],
        numLayers: gguf.metadata["llama.block_count"],
        numKeyValueHeads: gguf.metadata["llama.attention.head_count_kv"]!,
        headSize:
          gguf.metadata["llama.embedding_length"] /
          gguf.metadata["llama.attention.head_count"],
      };
    case "dots1" as string:
      return {
        hiddenSize: gguf.metadata["dots1.embedding_length"],
        numAttentionHeads: gguf.metadata["dots1.attention.head_count"],
        numLayers: gguf.metadata["dots1.block_count"],
        numKeyValueHeads: gguf.metadata["dots1.attention.head_count_kv"],
        headSize:
          gguf.metadata["dots1.embedding_length"] /
          gguf.metadata["dots1.attention.head_count"],
      };
    case "deepseek2":
      return {
        hiddenSize: gguf.metadata["deepseek2.embedding_length"],
        numAttentionHeads: gguf.metadata["deepseek2.attention.head_count"],
        numLayers: gguf.metadata["deepseek2.block_count"],
        numKeyValueHeads: gguf.metadata["deepseek2.attention.head_count_kv"]!,
        headSize:
          gguf.metadata["deepseek2.embedding_length"] /
          gguf.metadata["deepseek2.attention.head_count"],
      };
    case "hunyuan-moe" as string:
      return {
        hiddenSize: gguf.metadata["hunyuan-moe.embedding_length"],
        numAttentionHeads: gguf.metadata["hunyuan-moe.attention.head_count"],
        numLayers: gguf.metadata["hunyuan-moe.block_count"],
        numKeyValueHeads: gguf.metadata["hunyuan-moe.attention.head_count_kv"],
        headSize:
          gguf.metadata["hunyuan-moe.embedding_length"] /
          gguf.metadata["hunyuan-moe.attention.head_count"],
      };
    default: {
      Log.log("info", "Unknown architecture, attempting generic extraction. This may not work for all models.");
      const name = gguf.metadata["general.architecture"] as string;
      return {
        hiddenSize: gguf.metadata[`${name}.embedding_length`],
        numAttentionHeads: gguf.metadata[`${name}.attention.head_count`],
        numLayers: gguf.metadata[`${name}.block_count`],
        numKeyValueHeads: gguf.metadata[`${name}.attention.head_count_kv`]!,
        headSize:
          gguf.metadata[`${name}.embedding_length`] /
          gguf.metadata[`${name}.attention.head_count`],
      };
    }

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
  gpuPercentage,
}: {
  gguf: GGUFParseOutput;
  gpus: Gpu[];
  ramBytes: number;
  contextLength: number;
  contextQuantizationSize: number;
  gpuPercentage: number;
}): boolean {
  const kvSize = calculateKvCacheSizeBytes(
    gguf,
    contextLength,
    contextQuantizationSize
  );
  const tensorSize = calculateTensorsSizeBytes(gguf);
  const totalModelSize = kvSize + tensorSize;
  const totalGpuMemory =
    gpus.reduce((acc, gpu) => acc + gpu.memoryTotalBytes, 0) * gpuPercentage;
  const totalRamMemory = ramBytes;
  const totalMemory = totalGpuMemory + totalRamMemory;
  return totalModelSize <= totalMemory;
}

class Device {
  public readonly name: string;
  public readonly memoryTotalBytes: number;
  public readonly priority: number;
  public bytesAllocated: number = 0;
  public utilizationPercentage: number = 0.9;
  private unsafe: boolean = false;

  constructor(
    name: string,
    memoryTotalBytes: number,
    priority: number,
    gpuPercentage
  ) {
    this.name = name;
    this.memoryTotalBytes = memoryTotalBytes;
    this.priority = priority;
    this.utilizationPercentage = gpuPercentage;
  }

  private get safeMemoryTotalBytes(): number {
    return this.memoryTotalBytes * this.utilizationPercentage;
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

  public allocateOnDevice(
    deviceName: string,
    requiredMemoryBytes: number,
    tensorName?: string
  ): Device {
    const device = this.devices.find((d) => d.name === deviceName);
    if (!device) {
      throw new Error(`Device ${deviceName} not found.`);
    }
    if (!device.canAllocate(requiredMemoryBytes)) {
      throw new Error(
        `Cannot allocate ${bytesToMiB(requiredMemoryBytes).toFixed(
          2
        )} MiB on device ${deviceName}.`
      );
    }
    device.alloc(requiredMemoryBytes);
    if (tensorName) {
      this.tensorMap[tensorName] = device.name;
    }
    return device;
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
  gpuPercentage = 0.9,
  granularGpuPercentage,
}: {
  gguf: GGUFParseOutput;
  gpus: Gpu[];
  ramBytes: number;
  contextLength: number;
  contextQuantizationSize: number;
  check: boolean;
  gpuPercentage?: number;
  granularGpuPercentage?: number[];
}) {
  if (
    check &&
    !modelFitsInMemory({
      gguf,
      gpus,
      ramBytes,
      contextLength,
      contextQuantizationSize,
      gpuPercentage,
    })
  ) {
    throw new Error(
      "Model does not fit in combined GPU and RAM memory. Try reducing context length or quantization size."
    );
  }
  const metadata = extractMetadata(gguf);
  const cpuDevice = new Device("CPU", ramBytes, 0, 1);
  if (!check) {
    cpuDevice.setUnsafe();
  }
  const gpuDevices = gpus.map(
    (gpu) =>
      new Device(
        `CUDA${gpu.cudaId}`,
        gpu.memoryTotalBytes,
        gpu.memoryTotalBytes,
        gpuPercentage
      ) // Use GPU memory as priority as a heuristic for computation power
  );
  if (granularGpuPercentage) {
    if (granularGpuPercentage.length !== gpuDevices.length) {
      throw new Error(
        `Granular GPU percentages length (${granularGpuPercentage.length}) does not match number of GPUs (${gpuDevices.length}).`
      );
    }
    for (let i = 0; i < gpuDevices.length; i++) {
      const cudaId = gpus[i].cudaId;
      const device = gpuDevices.find((d) => d.name === `CUDA${cudaId}`);
      if (device) {
        device.utilizationPercentage = granularGpuPercentage[i];
      }
    }
  }

  const allocator = new DeviceAllocator([cpuDevice, ...gpuDevices]);
  const seen = new Set<string>();

  // allocate the emedding tensor to CPU
  // this is because llama.cpp doesn't support some quantization types on GPU for embedding tensors
  const embeddingTensor = gguf.tensorInfos.find(
    (tensor) => tensor.name === "token_embd.weight"
  );
  if (embeddingTensor) {
    const embeddingSize = calculateTensorSizeBytes(embeddingTensor);
    allocator.allocateOnDevice("CPU", embeddingSize, embeddingTensor.name);
    seen.add(embeddingTensor.name);
    Log.log(
      "info",
      `Embedding tensor ${embeddingTensor.name} allocated on CPU: ${bytesToMiB(
        embeddingSize
      ).toFixed(2)} MiB`
    );
  }

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
  for (const tensor of gguf.tensorInfos) {
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
  for (const tensor of gguf.tensorInfos) {
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
  for (const tensor of gguf.tensorInfos) {
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
  for (const tensor of gguf.tensorInfos) {
    if (seen.has(tensor.name)) {
      continue;
    }
    const tensorSize = calculateTensorSizeBytes(tensor);
    allocator.allocate(tensorSize, tensor.name);
    totalRestBytes += tensorSize;
    seen.add(tensor.name);
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
  command = command.trim();
  Log.log("default", command);
  return {
    command,
    tensorMap: allocator.tensorMap,
    deviceAllocation: allocator.devices.map((device) => ({
      name: device.name,
      bytesAllocated: device.bytesAllocated,
      memoryTotalBytes: device.memoryTotalBytes,
      utilizationPercentage: device.utilizationPercentage,
    })),
  };
}
