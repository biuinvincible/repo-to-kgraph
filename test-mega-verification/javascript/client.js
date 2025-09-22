/**
 * JavaScript client with modern ES6+ patterns.
 *
 * Expected entities: 12
 * - Classes: ApiClient, AuthManager
 * - Functions: createClient, validateToken, refreshAuthToken
 * - Methods: request, authenticate, logout, isAuthenticated, etc.
 * - Arrow functions and async patterns
 */

import { EventEmitter } from 'events';
import axios from 'axios';

// Constants and configuration
const API_BASE_URL = 'https://api.example.com';
const DEFAULT_TIMEOUT = 30000;
const MAX_RETRY_ATTEMPTS = 3;

/**
 * Authentication manager with token handling.
 */
class AuthManager extends EventEmitter {
    constructor(clientId, clientSecret) {
        super();
        this.clientId = clientId;
        this.clientSecret = clientSecret;
        this.accessToken = null;
        this.refreshToken = null;
        this.tokenExpiry = null;
        this._refreshPromise = null;
    }

    /**
     * Authenticate with credentials and get tokens.
     */
    async authenticate(username, password) {
        try {
            const response = await axios.post('/auth/login', {
                username,
                password,
                client_id: this.clientId,
                client_secret: this.clientSecret
            });

            const { access_token, refresh_token, expires_in } = response.data;

            this.accessToken = access_token;
            this.refreshToken = refresh_token;
            this.tokenExpiry = Date.now() + (expires_in * 1000);

            this.emit('authenticated', {
                username,
                expiresIn: expires_in
            });

            return true;
        } catch (error) {
            this.emit('authentication_failed', { error: error.message });
            throw new Error(`Authentication failed: ${error.message}`);
        }
    }

    /**
     * Check if user is currently authenticated.
     */
    isAuthenticated() {
        return this.accessToken && Date.now() < this.tokenExpiry;
    }

    /**
     * Refresh the access token using refresh token.
     */
    async refreshAuthToken() {
        if (!this.refreshToken) {
            throw new Error('No refresh token available');
        }

        // Prevent concurrent refresh attempts
        if (this._refreshPromise) {
            return this._refreshPromise;
        }

        this._refreshPromise = this._performTokenRefresh();

        try {
            return await this._refreshPromise;
        } finally {
            this._refreshPromise = null;
        }
    }

    async _performTokenRefresh() {
        const response = await axios.post('/auth/refresh', {
            refresh_token: this.refreshToken,
            client_id: this.clientId,
            client_secret: this.clientSecret
        });

        const { access_token, expires_in } = response.data;

        this.accessToken = access_token;
        this.tokenExpiry = Date.now() + (expires_in * 1000);

        this.emit('token_refreshed', { expiresIn: expires_in });

        return access_token;
    }

    /**
     * Logout and clear tokens.
     */
    async logout() {
        try {
            if (this.accessToken) {
                await axios.post('/auth/logout', {}, {
                    headers: { Authorization: `Bearer ${this.accessToken}` }
                });
            }
        } catch (error) {
            console.warn('Logout request failed:', error.message);
        } finally {
            this.accessToken = null;
            this.refreshToken = null;
            this.tokenExpiry = null;
            this.emit('logged_out');
        }
    }

    /**
     * Get current access token, refreshing if needed.
     */
    async getValidToken() {
        if (this.isAuthenticated()) {
            return this.accessToken;
        }

        if (this.refreshToken) {
            return await this.refreshAuthToken();
        }

        throw new Error('No valid authentication available');
    }
}

/**
 * API client with authentication and retry logic.
 */
class ApiClient {
    constructor(baseUrl = API_BASE_URL, timeout = DEFAULT_TIMEOUT) {
        this.baseUrl = baseUrl;
        this.timeout = timeout;
        this.authManager = null;
        this.requestInterceptors = [];
        this.responseInterceptors = [];

        // Create axios instance
        this.httpClient = axios.create({
            baseURL: this.baseUrl,
            timeout: this.timeout
        });

        this._setupInterceptors();
    }

    /**
     * Set authentication manager for the client.
     */
    setAuthManager(authManager) {
        this.authManager = authManager;

        // Listen to auth events
        authManager.on('logged_out', () => {
            this.emit('authentication_lost');
        });

        return this;
    }

    /**
     * Setup request and response interceptors.
     */
    _setupInterceptors() {
        // Request interceptor for authentication
        this.httpClient.interceptors.request.use(
            async (config) => {
                if (this.authManager) {
                    try {
                        const token = await this.authManager.getValidToken();
                        config.headers.Authorization = `Bearer ${token}`;
                    } catch (error) {
                        console.warn('Failed to get auth token:', error.message);
                    }
                }

                // Apply custom request interceptors
                for (const interceptor of this.requestInterceptors) {
                    config = await interceptor(config);
                }

                return config;
            },
            (error) => Promise.reject(error)
        );

        // Response interceptor for error handling
        this.httpClient.interceptors.response.use(
            (response) => {
                // Apply custom response interceptors
                for (const interceptor of this.responseInterceptors) {
                    response = interceptor(response);
                }
                return response;
            },
            async (error) => {
                if (error.response?.status === 401 && this.authManager) {
                    try {
                        await this.authManager.refreshAuthToken();
                        // Retry original request
                        return this.httpClient(error.config);
                    } catch (refreshError) {
                        this.authManager.emit('authentication_expired');
                    }
                }
                return Promise.reject(error);
            }
        );
    }

    /**
     * Make HTTP request with retry logic.
     */
    async request(method, url, data = null, options = {}) {
        const config = {
            method,
            url,
            data,
            ...options
        };

        let lastError;

        for (let attempt = 1; attempt <= MAX_RETRY_ATTEMPTS; attempt++) {
            try {
                const response = await this.httpClient(config);
                return response.data;
            } catch (error) {
                lastError = error;

                // Don't retry on client errors (4xx)
                if (error.response?.status >= 400 && error.response?.status < 500) {
                    break;
                }

                // Wait before retry with exponential backoff
                if (attempt < MAX_RETRY_ATTEMPTS) {
                    const delay = Math.pow(2, attempt) * 1000;
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }

        throw lastError;
    }

    /**
     * Convenience methods for HTTP verbs.
     */
    async get(url, options = {}) {
        return this.request('GET', url, null, options);
    }

    async post(url, data, options = {}) {
        return this.request('POST', url, data, options);
    }

    async put(url, data, options = {}) {
        return this.request('PUT', url, data, options);
    }

    async delete(url, options = {}) {
        return this.request('DELETE', url, null, options);
    }

    /**
     * Add custom request interceptor.
     */
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
        return this;
    }

    /**
     * Add custom response interceptor.
     */
    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
        return this;
    }
}

// Utility functions

/**
 * Validate JWT token structure.
 */
export const validateToken = (token) => {
    if (!token || typeof token !== 'string') {
        return false;
    }

    const parts = token.split('.');
    return parts.length === 3;
};

/**
 * Factory function to create configured API client.
 */
export const createClient = (config = {}) => {
    const {
        baseUrl = API_BASE_URL,
        timeout = DEFAULT_TIMEOUT,
        clientId,
        clientSecret
    } = config;

    const client = new ApiClient(baseUrl, timeout);

    if (clientId && clientSecret) {
        const authManager = new AuthManager(clientId, clientSecret);
        client.setAuthManager(authManager);
    }

    return client;
};

/**
 * Create client with default configuration.
 */
export const createDefaultClient = () => {
    return createClient({
        clientId: process.env.API_CLIENT_ID,
        clientSecret: process.env.API_CLIENT_SECRET
    });
};

// Export classes and utilities
export { ApiClient, AuthManager };
export default createDefaultClient;