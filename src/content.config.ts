import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '*.md', base: './blog' }),
  schema: z.object({}).passthrough(),
});

export const collections = { posts };
