import { Command } from "commander";
import { getNvidiaGpus } from "./nvidia.ts";
import optimize from "./optimize.ts";
import getRamBytes from "./system.ts";
import { downloadGguf } from "./gguf.ts";
import Log from "./log.ts";

const program = new Command();
program
  .name("nvidia-gguf-optimizer")
  .description("Optimize GGUF files for NVIDIA GPUs")
  .version("1.0.0");
program
  .option("-g, --gguf-url <url>", "URL of the GGUF file to optimize")
  .option("-c, --context-length <length>", "Context length for optimization")
  .option("--context-quantization-size <size>", "Context quantization size")
  .option(
    "--no-check",
    "Skip system resource limits check. Useful for when you're using swap"
  )
  .option(
    "--gpu-percentage <percentage>",
    "Percentage of GPU memory to use for allocation. Default is 0.9"
  )
  .option(
    "--granular-gpu-percentage <percentage>",
    'Set the percentage of GPU for each GPU. Should be formatted like "0.9,0.8,0.7" where the index of the percentage corresponds to the CUDA device'
  )
  .option("--verbose", "Enable verbose logging")
  .action(async (options) => {
    if (options.verbose) {
      Log.vebose = true;
    }
    let ggufUrl = options.ggufUrl;
    if (!ggufUrl) {
      Log.error("GGUF URL is required. Please provide a valid URL.");
      process.exit(1);
    }
    let contextLength = options.contextLength;
    if (!contextLength) {
      Log.error("Context length is required. Please provide a valid length.");
      process.exit(1);
    }
    let contextQuantizationSize = options.contextQuantizationSize || "16";
    if (
      !contextQuantizationSize ||
      !["4", "8", "16"].includes(contextQuantizationSize)
    ) {
      Log.error(
        "Context quantization size must be one of 4, 8, or 16. Please provide a valid size."
      );
      process.exit(1);
    }
    if (options.granularGpuPercentage && options.gpuPercentage) {
      Log.error(
        "You cannot use both --gpu-percentage and --granular-gpu-percentage options at the same time. Please choose one."
      );
      process.exit(1);
    }
    let granularGpuPercentage: number[] | undefined;
    if (options.granularGpuPercentage) {
      granularGpuPercentage = options.granularGpuPercentage
        .split(",")
        .map((p: string) => {
          const percentage = parseFloat(p);
          if (isNaN(percentage) || percentage < 0 || percentage > 1) {
            Log.error(
              `Invalid GPU percentage: ${p}. It should be a number between 0 and 1.`
            );
            process.exit(1);
          }
          return percentage;
        });
    }
    optimize({
      gguf: await downloadGguf(ggufUrl),
      gpus: getNvidiaGpus(),
      ramBytes: getRamBytes(),
      contextLength: +contextLength,
      contextQuantizationSize: +contextQuantizationSize,
      check: options.check,
      gpuPercentage: options.gpuPercentage
        ? parseFloat(options.gpuPercentage)
        : undefined,
      granularGpuPercentage,
    });
  });
await program.parseAsync(process.argv);
