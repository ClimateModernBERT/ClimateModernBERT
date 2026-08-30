/**
 * GitHub Pages serves this site from /ClimateModernBERT/, not from /.
 * Never write a bare absolute path — route it through here.
 */
const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");

export function withBase(path: string): string {
  if (/^(https?:)?\/\//.test(path) || path.startsWith("#") || path.startsWith("mailto:")) {
    return path;
  }
  return `${BASE}/${path.replace(/^\//, "")}`;
}
