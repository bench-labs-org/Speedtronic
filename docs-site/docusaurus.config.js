// @ts-check

const {themes: prismThemes} = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Speedtronic',
  tagline: 'GPU-agnostic PyTorch training, documented from source',
  url: 'https://speedtronic-docs.pages.dev',
  baseUrl: '/',
  organizationName: 'bench-labs-org',
  projectName: 'Speedtronic',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'warn',
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: 'docs',
          sidebarPath: require.resolve('./sidebars.js'),
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Speedtronic',
      items: [
        {type: 'docSidebar', sidebarId: 'documentation', position: 'left', label: 'Documentation'},
        {to: '/docs/reference/source-inventory', label: 'Source inventory', position: 'left'},
        {href: 'https://github.com/bench-labs-org/Speedtronic', label: 'Repository', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Quickstart', to: '/docs/getting-started/quickstart'},
            {label: 'Architecture', to: '/docs/reference/architecture'},
            {label: 'Configuration', to: '/docs/reference/configuration'},
          ],
        },
        {
          title: 'Advanced',
          items: [
            {label: 'DumbDiLoCo', to: '/docs/distributed/overview'},
            {label: 'Security and limits', to: '/docs/operations/security-and-limitations'},
            {label: 'Test map', to: '/docs/appendix/test-map'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'Trainer', to: '/docs/reference/runtime-and-trainer'},
            {label: 'Data API', to: '/docs/reference/data'},
            {label: 'Module inventory', to: '/docs/reference/source-inventory'},
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} BenchLabs. Apache-2.0. Cloudflare Pages documentation.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'yaml', 'python', 'toml'],
    },
  },
};

module.exports = config;
