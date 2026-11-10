import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const target = process.env.VITE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': target,
      '/model-info': target,
      '/predict': target,
      '/audit': target,
    },
  },
})
