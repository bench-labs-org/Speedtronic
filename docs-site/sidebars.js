// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  documentation: [
    'index',
    {
      type: 'category',
      label: 'Getting started',
      collapsed: false,
      items: ['getting-started/quickstart', 'getting-started/core-concepts'],
    },
    {
      type: 'category',
      label: 'Tutorials',
      collapsed: false,
      items: [
        'tutorials/custom-model',
        'tutorials/custom-data',
        'tutorials/local-performance',
        'tutorials/checkpoint-resume',
        'tutorials/observability',
      ],
    },
    {
      type: 'category',
      label: 'Core reference',
      collapsed: false,
      items: [
        'reference/architecture',
        'reference/cli',
        'reference/configuration',
        'reference/runtime-and-trainer',
        'reference/data',
        'reference/reference-model',
        'reference/precision',
        'reference/checkpointing',
        'reference/observability',
        'reference/extensions',
      ],
    },
    {
      type: 'category',
      label: 'Speedtronic v2',
      collapsed: false,
      items: [
        'v2/index',
        'v2/optimizers',
        'v2/scheduling',
        'v2/shape-validation',
        'v2/migration',
        'v2/roadmap',
      ],
    },
    {
      type: 'category',
      label: 'DumbDiLoCo',
      collapsed: false,
      items: [
        'distributed/overview',
        'distributed/coordinator',
        'distributed/hub-transport',
        'distributed/outer-loop',
        'distributed/tensor-format',
      ],
    },
    {
      type: 'category',
      label: 'Examples and operations',
      collapsed: false,
      items: [
        'examples/shipped-configs',
        'examples/shipped-examples',
        'operations/testing',
        'operations/packaging',
        'operations/troubleshooting',
        'operations/security-and-limitations',
      ],
    },
    {
      type: 'category',
      label: 'Appendix',
      collapsed: false,
      items: [
        'reference/source-inventory',
        'reference/generated-source-inventory',
        'appendix/test-map',
        'appendix/glossary',
      ],
    },
  ],
};

module.exports = sidebars;
