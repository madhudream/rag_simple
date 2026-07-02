import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://rag-simple.pages.dev',
  markdown: {
    shikiConfig: {
      theme: 'vesper',
      wrap: false,
    },
  },
});
