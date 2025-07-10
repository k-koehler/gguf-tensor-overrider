import type { Gpu } from "../src/nvidia.ts";

export function assertEqual(
  actual: any,
  expected: any,
  message?: string
): void {
  if (actual !== expected) {
    throw new Error(message || `Expected ${expected}, but got ${actual}`);
  }
}

export function baselineGpus(): Gpu[] {
  return [
    {
      name: "NVIDIA GeForce RTX 3090",
      memoryTotalBytes: 24 * 1024 * 1024 * 1024,
      cudaId: 0,
    },
    {
      name: "NVIDIA GeForce RTX 3090",
      memoryTotalBytes: 24 * 1024 * 1024 * 1024,
      cudaId: 1,
    },
    {
      name: "NVIDIA GeForce RTX 3060",
      memoryTotalBytes: 12 * 1024 * 1024 * 1024,
      cudaId: 2,
    },
  ];
}

export function baselineRamBytes(): number {
  return 128 * 1024 * 1024 * 1024;
}

export function baselineContextLength(): number {
  return 32768;
}

export function baselineContextQuantizationSize(): number {
  return 16;
}

export function ramGbBytes(gib: number) {
  return gib * 1024 * 1024 * 1024;
}
