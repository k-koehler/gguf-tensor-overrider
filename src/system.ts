import os from "os";

export default function getRamBytes(): number {
  return os.totalmem();
}
