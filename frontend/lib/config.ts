/**
 * Configuration utilities for environment variables
 * Handles both server-side and client-side environment access
 */

export const config = {
  /**
   * Get API URL with fallback logic
   * - Client-side: uses window location
   * - Server-side: uses environment variable
   */
  getApiUrl: (): string => {
    // Check if we're on the client side
    if (typeof window !== 'undefined') {
      // Client-side: construct URL from current location
      const protocol = window.location.protocol;
      const hostname = window.location.hostname;
      return `${protocol}//${hostname}:8000`;
    }
    
    // Server-side: use environment variable or fallback
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  },

  /**
   * Get Modal endpoint
   */
  getModalUrl: (): string => {
    return process.env.NEXT_PUBLIC_MODAL_ENDPOINT || 'https://your-modal-app.modal.run';
  },

  /**
   * Check if we're in development mode
   */
  isDevelopment: (): boolean => {
    return process.env.NODE_ENV === 'development';
  },

  /**
   * Get all API endpoints
   */
  getEndpoints: () => ({
    local: config.getApiUrl(),
    modal: config.getModalUrl(),
  }),
};

export default config;

