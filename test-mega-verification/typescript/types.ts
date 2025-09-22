/**
 * TypeScript file with advanced type definitions and patterns.
 *
 * Expected entities: 18+
 * - Interfaces: User, Repository, ServiceConfig, etc.
 * - Types: Status, EventCallback, etc.
 * - Classes: TypedEventEmitter, GenericRepository
 * - Generic functions with type constraints
 * - Decorators and metadata
 */

import { EventEmitter } from 'events';

// Type definitions and interfaces

export type Status = 'active' | 'inactive' | 'pending' | 'suspended';

export type EventCallback<T = any> = (data: T) => void | Promise<void>;

export interface User {
    id: number;
    email: string;
    name: string;
    status: Status;
    createdAt: Date;
    updatedAt: Date;
    metadata?: Record<string, unknown>;
}

export interface Repository<T, K = string> {
    findById(id: K): Promise<T | null>;
    findAll(filter?: Partial<T>): Promise<T[]>;
    create(entity: Omit<T, 'id' | 'createdAt' | 'updatedAt'>): Promise<T>;
    update(id: K, updates: Partial<T>): Promise<T>;
    delete(id: K): Promise<boolean>;
}

export interface ServiceConfig {
    readonly baseUrl: string;
    readonly timeout: number;
    readonly retryAttempts: number;
    readonly enableLogging: boolean;
    readonly apiKey?: string;
}

export interface PaginationOptions {
    page: number;
    limit: number;
    sortBy?: keyof User;
    sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResult<T> {
    data: T[];
    total: number;
    page: number;
    totalPages: number;
    hasNext: boolean;
    hasPrevious: boolean;
}

// Advanced generic types

export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type EventMap = {
    userCreated: User;
    userUpdated: { id: number; changes: Partial<User> };
    userDeleted: { id: number };
    error: Error;
};

// Decorator functions

export function deprecated(reason?: string) {
    return function <T extends { new (...args: any[]): {} }>(constructor: T) {
        return class extends constructor {
            constructor(...args: any[]) {
                console.warn(`Warning: ${constructor.name} is deprecated. ${reason || ''}`);
                super(...args);
            }
        };
    };
}

export function log(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = function (...args: any[]) {
        console.log(`Calling ${propertyName} with args:`, args);
        const result = method.apply(this, args);
        console.log(`${propertyName} returned:`, result);
        return result;
    };
}

export function validate<T>(validator: (value: T) => boolean, message: string) {
    return function (target: any, propertyName: string, descriptor: PropertyDescriptor) {
        const method = descriptor.value;

        descriptor.value = function (value: T, ...args: any[]) {
            if (!validator(value)) {
                throw new Error(`Validation failed for ${propertyName}: ${message}`);
            }
            return method.apply(this, [value, ...args]);
        };
    };
}

// Generic classes with complex type relationships

export class TypedEventEmitter<TEvents extends Record<string, any> = EventMap> {
    private emitter = new EventEmitter();
    private listeners: Map<keyof TEvents, Set<EventCallback>> = new Map();

    on<K extends keyof TEvents>(event: K, callback: EventCallback<TEvents[K]>): this {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(callback);

        this.emitter.on(event as string, callback);
        return this;
    }

    off<K extends keyof TEvents>(event: K, callback: EventCallback<TEvents[K]>): this {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.delete(callback);
            this.emitter.off(event as string, callback);
        }
        return this;
    }

    emit<K extends keyof TEvents>(event: K, data: TEvents[K]): boolean {
        return this.emitter.emit(event as string, data);
    }

    once<K extends keyof TEvents>(event: K, callback: EventCallback<TEvents[K]>): this {
        const wrapper: EventCallback<TEvents[K]> = (data) => {
            callback(data);
            this.off(event, wrapper);
        };
        return this.on(event, wrapper);
    }

    removeAllListeners<K extends keyof TEvents>(event?: K): this {
        if (event) {
            this.listeners.delete(event);
            this.emitter.removeAllListeners(event as string);
        } else {
            this.listeners.clear();
            this.emitter.removeAllListeners();
        }
        return this;
    }

    getListenerCount<K extends keyof TEvents>(event: K): number {
        return this.listeners.get(event)?.size || 0;
    }
}

// Abstract base repository with generics

export abstract class GenericRepository<T, K = string> implements Repository<T, K> {
    protected items: Map<K, T> = new Map();
    protected eventEmitter = new TypedEventEmitter<EventMap>();

    constructor(protected config: ServiceConfig) {
        this.validateConfig(config);
    }

    @log
    @validate<ServiceConfig>(
        (config) => config.timeout > 0,
        'Timeout must be greater than 0'
    )
    protected validateConfig(config: ServiceConfig): void {
        if (!config.baseUrl) {
            throw new Error('Base URL is required');
        }
    }

    abstract generateId(): K;
    abstract validateEntity(entity: Partial<T>): boolean;
    abstract createTimestamps(): Pick<T, 'createdAt' | 'updatedAt'> & any;

    async findById(id: K): Promise<T | null> {
        this.logOperation('findById', { id });
        return this.items.get(id) || null;
    }

    async findAll(filter?: Partial<T>): Promise<T[]> {
        this.logOperation('findAll', { filter });
        const allItems = Array.from(this.items.values());

        if (!filter) {
            return allItems;
        }

        return allItems.filter(item =>
            Object.entries(filter).every(([key, value]) =>
                (item as any)[key] === value
            )
        );
    }

    async create(entityData: Omit<T, 'id' | 'createdAt' | 'updatedAt'>): Promise<T> {
        if (!this.validateEntity(entityData)) {
            throw new Error('Invalid entity data');
        }

        const id = this.generateId();
        const timestamps = this.createTimestamps();
        const entity = { ...entityData, id, ...timestamps } as T;

        this.items.set(id, entity);

        // Type-safe event emission
        this.eventEmitter.emit('userCreated', entity as any);

        this.logOperation('create', { id, entity });
        return entity;
    }

    async update(id: K, updates: Partial<T>): Promise<T> {
        const existing = await this.findById(id);
        if (!existing) {
            throw new Error(`Entity with id ${id} not found`);
        }

        const updated = {
            ...existing,
            ...updates,
            updatedAt: new Date()
        } as T;

        if (!this.validateEntity(updated)) {
            throw new Error('Invalid update data');
        }

        this.items.set(id, updated);

        // Type-safe event emission
        this.eventEmitter.emit('userUpdated', { id: id as any, changes: updates });

        this.logOperation('update', { id, updates, entity: updated });
        return updated;
    }

    async delete(id: K): Promise<boolean> {
        const existed = this.items.has(id);
        if (existed) {
            this.items.delete(id);
            this.eventEmitter.emit('userDeleted', { id: id as any });
            this.logOperation('delete', { id, success: true });
        }
        return existed;
    }

    async paginate(options: PaginationOptions): Promise<PaginatedResult<T>> {
        const { page, limit, sortBy, sortOrder = 'asc' } = options;
        let items = Array.from(this.items.values());

        // Sort if specified
        if (sortBy) {
            items.sort((a, b) => {
                const aVal = (a as any)[sortBy];
                const bVal = (b as any)[sortBy];
                const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
                return sortOrder === 'desc' ? -comparison : comparison;
            });
        }

        const total = items.length;
        const totalPages = Math.ceil(total / limit);
        const startIndex = (page - 1) * limit;
        const endIndex = startIndex + limit;
        const data = items.slice(startIndex, endIndex);

        return {
            data,
            total,
            page,
            totalPages,
            hasNext: page < totalPages,
            hasPrevious: page > 1
        };
    }

    on<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
        this.eventEmitter.on(event, callback);
        return this;
    }

    off<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
        this.eventEmitter.off(event, callback);
        return this;
    }

    protected logOperation(operation: string, data: any): void {
        if (this.config.enableLogging) {
            console.log(`[${this.constructor.name}] ${operation}:`, data);
        }
    }
}

// Concrete implementation

@deprecated('Use UserRepositoryV2 instead')
export class UserRepository extends GenericRepository<User, number> {
    private idCounter = 1;

    constructor(config: ServiceConfig) {
        super(config);
    }

    generateId(): number {
        return this.idCounter++;
    }

    validateEntity(entity: Partial<User>): boolean {
        return !!(
            entity.email &&
            entity.name &&
            typeof entity.email === 'string' &&
            typeof entity.name === 'string' &&
            entity.email.includes('@')
        );
    }

    createTimestamps(): Pick<User, 'createdAt' | 'updatedAt'> {
        const now = new Date();
        return {
            createdAt: now,
            updatedAt: now
        };
    }

    // Additional user-specific methods
    async findByEmail(email: string): Promise<User | null> {
        const users = await this.findAll({ email } as Partial<User>);
        return users.length > 0 ? users[0] : null;
    }

    async findActiveUsers(): Promise<User[]> {
        return this.findAll({ status: 'active' } as Partial<User>);
    }

    async updateStatus(id: number, status: Status): Promise<User> {
        return this.update(id, { status, updatedAt: new Date() } as Partial<User>);
    }
}

// Factory functions with complex type relationships

export function createRepository<T, K = string>(
    config: ServiceConfig,
    factory: new (config: ServiceConfig) => GenericRepository<T, K>
): GenericRepository<T, K> {
    return new factory(config);
}

export function createUserRepository(config?: Partial<ServiceConfig>): UserRepository {
    const defaultConfig: ServiceConfig = {
        baseUrl: 'http://localhost:3000',
        timeout: 5000,
        retryAttempts: 3,
        enableLogging: false,
        ...config
    };

    return new UserRepository(defaultConfig);
}

// Utility type functions

export function isUser(obj: any): obj is User {
    return (
        typeof obj === 'object' &&
        obj !== null &&
        typeof obj.id === 'number' &&
        typeof obj.email === 'string' &&
        typeof obj.name === 'string' &&
        ['active', 'inactive', 'pending', 'suspended'].includes(obj.status)
    );
}

export function ensureArray<T>(value: T | T[]): T[] {
    return Array.isArray(value) ? value : [value];
}

export async function withRetry<T>(
    operation: () => Promise<T>,
    maxAttempts: number = 3,
    delay: number = 1000
): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error as Error;

            if (attempt === maxAttempts) {
                break;
            }

            await new Promise(resolve => setTimeout(resolve, delay * attempt));
        }
    }

    throw lastError!;
}