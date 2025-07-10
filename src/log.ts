export default class Log {
  public static vebose: boolean = false;
  public static noLog: boolean = false;

  public static log(logLevel: "info" | "default" | "nolog", message: string) {
    if (Log.noLog) {
      return;
    }
    if (logLevel === "info" && Log.vebose) {
      console.log(`[${logLevel.toUpperCase()}] ${message}`);
    } else if (logLevel === "default") {
      console.log(message);
    }
  }

  public static warn(message: string) {
    if (Log.noLog) {
      return;
    }
    console.warn(`[WARN] ${message}`);
  }

  public static error(message: string) {
    if (Log.noLog) {
      return;
    }
    console.error(`[ERROR] ${message}`);
  }
}
