import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "fxlin's Blog",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: false,
    analytics: null,
    locale: "zh-CN",
    baseUrl: "fxlin0261.github.io",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: {
          name: "Noto Sans SC",
          weights: [500, 700],
          includeItalic: false,
        },
        body: {
          name: "Noto Sans SC",
          weights: [400, 500],
          includeItalic: false,
        },
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#fbfaf7",
          lightgray: "#e7e2da",
          gray: "#b7aea1",
          darkgray: "#5f5951",
          dark: "#24211d",
          secondary: "#0f4c81",
          tertiary: "#7b8b98",
          highlight: "rgba(15, 76, 129, 0.12)",
          textHighlight: "#ffe08a99",
        },
        darkMode: {
          light: "#14171b",
          lightgray: "#2a3037",
          gray: "#69717a",
          darkgray: "#d8dde2",
          dark: "#edf2f7",
          secondary: "#7fb2e5",
          tertiary: "#98a7b4",
          highlight: "rgba(127, 178, 229, 0.14)",
          textHighlight: "#8c6a0099",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
