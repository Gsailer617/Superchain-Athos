import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import crypto from 'crypto';
import { promisify } from 'util';

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

interface DatabaseConfig {
    dataDir: string;
    backupDir: string;
    compressionLevel?: number;
    maxBackups?: number;
    indexedFields?: string[];
}

export interface BackupMetadata {
    timestamp: number;
    hash: string;
    size: number;
    compressionRatio: number;
    version: string;
}

interface DataIndex {
    [field: string]: {
        [value: string]: number[];  // Array of record indices
    };
}

export class DatabaseManager {
    private readonly dataDir: string;
    private readonly backupDir: string;
    private readonly compressionLevel: number;
    private readonly maxBackups: number;
    private readonly indexedFields: Set<string>;
    private indices: { [collection: string]: DataIndex } = {};
    private dataCache: { [collection: string]: any[] } = {};
    private readonly DB_VERSION = '1.0.0';

    constructor(config: DatabaseConfig) {
        this.dataDir = config.dataDir;
        this.backupDir = config.backupDir;
        this.compressionLevel = config.compressionLevel || 6;
        this.maxBackups = config.maxBackups || 5;
        this.indexedFields = new Set(config.indexedFields || []);

        this.ensureDirectories();
        this.initializeIndices();
    }

    private ensureDirectories() {
        [this.dataDir, this.backupDir].forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
        });
    }

    private async initializeIndices() {
        const collections = await this.listCollections();
        for (const collection of collections) {
            await this.loadCollection(collection);
            this.buildIndices(collection);
        }
    }

    private buildIndices(collection: string) {
        const data = this.dataCache[collection] || [];
        const indices: DataIndex = {};

        this.indexedFields.forEach(field => {
            indices[field] = {};
            data.forEach((record, index) => {
                const value = String(record[field]);
                if (!indices[field][value]) {
                    indices[field][value] = [];
                }
                indices[field][value].push(index);
            });
        });

        this.indices[collection] = indices;
    }

    async saveCollection(collection: string, data: any[]): Promise<void> {
        const filePath = path.join(this.dataDir, `${collection}.json.gz`);
        const jsonData = JSON.stringify(data);
        const compressed = await gzip(jsonData, { level: this.compressionLevel });
        
        await fs.promises.writeFile(filePath, compressed);
        this.dataCache[collection] = data;
        this.buildIndices(collection);
    }

    async loadCollection(collection: string): Promise<any[]> {
        if (this.dataCache[collection]) {
            return this.dataCache[collection];
        }

        const filePath = path.join(this.dataDir, `${collection}.json.gz`);
        if (!fs.existsSync(filePath)) {
                return [];
            }

        const compressed = await fs.promises.readFile(filePath);
        const decompressed = await gunzip(compressed);
        const data = JSON.parse(decompressed.toString());
        
        this.dataCache[collection] = data;
        return data;
        }

    async query(collection: string, conditions: { [field: string]: any }): Promise<any[]> {
        const data = await this.loadCollection(collection);
        let resultIndices = new Set<number>();
        let isFirstCondition = true;

        for (const [field, value] of Object.entries(conditions)) {
            if (this.indexedFields.has(field)) {
                // Use index for this field
                const matchingIndices = this.indices[collection]?.[field]?.[String(value)] || [];
                if (isFirstCondition) {
                    matchingIndices.forEach(i => resultIndices.add(i));
                    isFirstCondition = false;
                } else {
                    resultIndices = new Set([...resultIndices].filter(i => matchingIndices.includes(i)));
    }
            } else {
                // Fallback to full scan for non-indexed fields
                const matches = data.reduce((acc, record, index) => {
                    if (record[field] === value) {
                        acc.add(index);
                    }
                    return acc;
                }, new Set<number>());

                if (isFirstCondition) {
                    resultIndices = matches;
                    isFirstCondition = false;
                } else {
                    resultIndices = new Set([...resultIndices].filter(i => matches.has(i)));
                }
            }
    }

        return Array.from(resultIndices).map(i => data[i]);
                }

    async createBackup(): Promise<BackupMetadata> {
        const timestamp = Date.now();
        const backupFileName = `backup_${timestamp}.tar.gz`;
        const backupPath = path.join(this.backupDir, backupFileName);
        const metadataPath = path.join(this.backupDir, 'backups.json');

        // Create backup archive
        const files = await fs.promises.readdir(this.dataDir);
        const archiveData = [];
        
        for (const file of files) {
            const filePath = path.join(this.dataDir, file);
            const content = await fs.promises.readFile(filePath);
            archiveData.push({
                name: file,
                content: content
        });
    }

        const archiveBuffer = Buffer.from(JSON.stringify(archiveData));
        const compressed = await gzip(archiveBuffer, { level: 9 });
        await fs.promises.writeFile(backupPath, compressed);
        
        // Calculate metadata
        const hash = crypto.createHash('sha256').update(compressed).digest('hex');
        const metadata: BackupMetadata = {
            timestamp,
            hash,
            size: compressed.length,
            compressionRatio: archiveBuffer.length / compressed.length,
            version: this.DB_VERSION
        };

        // Update backups list
        let backups: BackupMetadata[] = [];
        if (fs.existsSync(metadataPath)) {
            backups = JSON.parse(await fs.promises.readFile(metadataPath, 'utf8'));
        }
        backups.push(metadata);
        
        // Remove old backups if exceeding maxBackups
        while (backups.length > this.maxBackups) {
            const oldestBackup = backups.shift()!;
            const oldBackupPath = path.join(this.backupDir, `backup_${oldestBackup.timestamp}.tar.gz`);
            if (fs.existsSync(oldBackupPath)) {
                await fs.promises.unlink(oldBackupPath);
            }
        }

        await fs.promises.writeFile(metadataPath, JSON.stringify(backups, null, 2));
        return metadata;
    }

    async restoreFromBackup(timestamp: number): Promise<void> {
        const backupPath = path.join(this.backupDir, `backup_${timestamp}.tar.gz`);
        const metadataPath = path.join(this.backupDir, 'backups.json');

        if (!fs.existsSync(backupPath) || !fs.existsSync(metadataPath)) {
            throw new Error('Backup not found');
        }
        
        const backups: BackupMetadata[] = JSON.parse(
            await fs.promises.readFile(metadataPath, 'utf8')
        );
        const metadata = backups.find(b => b.timestamp === timestamp);
        if (!metadata) {
            throw new Error('Backup metadata not found');
        }
        
        // Read and decompress backup
        const compressed = await fs.promises.readFile(backupPath);
        const decompressed = await gunzip(compressed);
        const archiveData = JSON.parse(decompressed.toString());

        // Verify hash
        const hash = crypto.createHash('sha256').update(compressed).digest('hex');
        if (hash !== metadata.hash) {
            throw new Error('Backup integrity check failed');
        }
        
        // Clear current data directory
        const currentFiles = await fs.promises.readdir(this.dataDir);
        for (const file of currentFiles) {
            await fs.promises.unlink(path.join(this.dataDir, file));
        }

        // Restore files
        for (const file of archiveData) {
            await fs.promises.writeFile(
                path.join(this.dataDir, file.name),
                Buffer.from(file.content)
            );
    }

        // Reinitialize indices
        this.dataCache = {};
        await this.initializeIndices();
    }

    async migrateData(fromVersion: string, toVersion: string): Promise<void> {
        // Implement version-specific migration logic here
        const migrations = this.getMigrationPath(fromVersion, toVersion);
            
        for (const migration of migrations) {
            await this.executeMigration(migration);
        }
    }
            
    private getMigrationPath(fromVersion: string, toVersion: string): string[] {
        // Return array of migration steps needed
        // This is a placeholder - implement actual version path calculation
        return [`${fromVersion}_to_${toVersion}`];
        }

    private async executeMigration(migration: string): Promise<void> {
        // Implement specific migration logic here
        // This is a placeholder - implement actual migration logic
        console.log(`Executing migration: ${migration}`);
    }

    async listCollections(): Promise<string[]> {
        const files = await fs.promises.readdir(this.dataDir);
        return files
            .filter(f => f.endsWith('.json.gz'))
            .map(f => f.replace('.json.gz', ''));
    }
        
    async getBackupsList(): Promise<BackupMetadata[]> {
        const metadataPath = path.join(this.backupDir, 'backups.json');
        if (!fs.existsSync(metadataPath)) {
            return [];
            }
        return JSON.parse(await fs.promises.readFile(metadataPath, 'utf8'));
        }

    clearCache(): void {
        this.dataCache = {};
        this.indices = {};
    }
} 