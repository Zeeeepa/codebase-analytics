/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    // Make sure environment variables are available
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_MODAL_ENDPOINT: process.env.NEXT_PUBLIC_MODAL_ENDPOINT,
  },
  // Ensure proper handling of environment variables
  experimental: {
    // Enable server components
    serverComponentsExternalPackages: [],
  },
  // Configure webpack to handle environment variables properly
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Ensure process.env is available on client side for NEXT_PUBLIC_ variables
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    return config;
  },
  // Add async rewrites for API proxy
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
