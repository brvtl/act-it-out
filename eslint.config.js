import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'public/ort', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.strict,
  // Node scripts that also pass callbacks into the browser via Playwright's page.evaluate.
  { files: ['scripts/**/*.mjs'], languageOptions: { globals: { ...globals.node, ...globals.browser } } },
);
