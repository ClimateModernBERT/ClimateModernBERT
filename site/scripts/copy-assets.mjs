/**
 * Copies repository-owned assets into site/public/ before the build, so the PDF
 * and the framework figure live in exactly one place in git.
 *
 * Runs automatically via the `prebuild` npm script.
 */
import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(SITE);

const assets = [
  ["paper/climate-modernbert.pdf", "climate-modernbert.pdf"],
  ["main.png", "framework-overview.png"],
];

mkdirSync(join(SITE, "public"), { recursive: true });

for (const [from, to] of assets) {
  const src = join(ROOT, from);
  if (!existsSync(src)) {
    console.error(`copy-assets: missing ${from} — the site expects it at the repository root.`);
    process.exit(1);
  }
  copyFileSync(src, join(SITE, "public", to));
  console.log(`copy-assets: ${from} -> site/public/${to}`);
}
