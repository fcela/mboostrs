#!/usr/bin/env node

// Mermaid CLI appends --cssFile after laying out the graph. A custom web font
// therefore changes glyph widths after node dimensions have been fixed, which
// clips labels. This renderer loads the font before Mermaid measures any text,
// then embeds the same CSS in the standalone SVG.

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import puppeteer from "puppeteer";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const mermaidBundle = path.join(root, "node_modules/mermaid/dist/mermaid.min.js");

const [inputPath, outputPath, configPath, cssPath, rawScale = "1"] = process.argv.slice(2);
if (!inputPath || !outputPath || !configPath || !cssPath) {
  console.error(
    "usage: render_mermaid.mjs INPUT OUTPUT CONFIG CSS [DEVICE_SCALE_FACTOR]",
  );
  process.exit(2);
}

const [definition, css, rawConfig] = await Promise.all([
  fs.readFile(inputPath, "utf8"),
  fs.readFile(cssPath, "utf8"),
  fs.readFile(configPath, "utf8"),
]);
const config = JSON.parse(rawConfig);
const scale = Number(rawScale);
if (!Number.isFinite(scale) || scale <= 0) {
  throw new Error(`invalid device scale factor: ${rawScale}`);
}

const browser = await puppeteer.launch({ headless: "shell" });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 800, height: 600, deviceScaleFactor: scale });
  await page.setContent(
    `<!doctype html><html><head><style>${css}</style></head>` +
      '<body style="margin:0;background:transparent"><div id="container"></div></body></html>',
  );
  await page.addScriptTag({ path: mermaidBundle });
  await page.evaluate(
    async ({ definition, config, css }) => {
      await Promise.all([
        document.fonts.load('400 16px "DM Sans"'),
        document.fonts.load('700 16px "DM Sans"'),
      ]);
      await document.fonts.ready;

      globalThis.mermaid.initialize({ startOnLoad: false, ...config });
      const { svg } = await globalThis.mermaid.render("my-svg", definition);
      const container = document.getElementById("container");
      container.innerHTML = svg;
      const element = container.querySelector("svg");
      element.style.backgroundColor = "transparent";
      const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
      style.textContent = css;
      element.appendChild(style);
      await document.fonts.ready;
    },
    { definition, config, css },
  );

  if (path.extname(outputPath).toLowerCase() === ".svg") {
    const output = await page.$eval("svg", (svg) =>
      new XMLSerializer().serializeToString(svg),
    );
    await fs.writeFile(outputPath, output);
  } else if (path.extname(outputPath).toLowerCase() === ".png") {
    const clip = await page.$eval("svg", (svg) => {
      const bounds = svg.getBoundingClientRect();
      return {
        x: Math.floor(bounds.left),
        y: Math.floor(bounds.top),
        width: Math.ceil(bounds.width),
        height: Math.ceil(bounds.height),
      };
    });
    await page.setViewport({
      width: clip.x + clip.width,
      height: clip.y + clip.height,
      deviceScaleFactor: scale,
    });
    await page.screenshot({ path: outputPath, clip, omitBackground: true });
  } else {
    throw new Error(`unsupported output format: ${outputPath}`);
  }
} finally {
  await browser.close();
}
