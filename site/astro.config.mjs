// @ts-check
import { defineConfig } from "astro/config";

// The site is served from a repository subpath on GitHub Pages, never from "/".
// Every internal href and asset must go through withBase() in src/lib/base.ts.
export default defineConfig({
  // Published from the ClimateModernBERT org; `base` still matches the repo name.
  site: "https://climatemodernbert.github.io",
  base: "/ClimateModernBERT",
  trailingSlash: "ignore",
  build: { format: "directory" },
  compressHTML: true,
});
