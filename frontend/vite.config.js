import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:5000',
    //     changeOrigin: true,
    //   },
 
    // },
    proxy: {
      '/api': process.env.VITE_API_PROXY,
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
}) 
