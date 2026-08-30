// @ts-check
import { defineConfig } from "astro/config";

// The site is served from a repository subpath on GitHub Pages, never from "/".
// Every internal href and asset must go through withBase() in src/lib/base.ts.
export default defineConfig({
  site: "https://michaelyya.github.io",
  base: "/ClimateModernBERT",
  trailingSlash: "ignore",
  build: { format: "directory" },
  compressHTML: true,
});
