import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Download, CheckCircle2, Loader2, XCircle } from 'lucide-react';

interface Qwen3Status {
    artifactsPresent: boolean;
    modelsDir: string;
    missing: string[];
}

interface ProgressEvent {
    stage: string;
    received: number;
    total: number;
}

export function Qwen3ModelManager() {
    const [status, setStatus] = useState<Qwen3Status | null>(null);
    const [downloading, setDownloading] = useState(false);
    const [progress, setProgress] = useState<{ stage: string; pct: number } | null>(null);
    const [error, setError] = useState<string | null>(null);

    const refresh = async () => {
        try {
            setStatus(await invoke<Qwen3Status>('qwen3_status'));
        } catch (e) {
            setError(String(e));
        }
    };

    useEffect(() => {
        void refresh();
        let unlisten: (() => void) | undefined;
        void import('@tauri-apps/api/event').then(({ listen }) =>
            listen<ProgressEvent>('qwen3-download-progress', (ev) => {
                const { stage, received, total } = ev.payload;
                setProgress({ stage, pct: total > 0 ? Math.round((received / total) * 100) : -1 });
            }),
        ).then((fn) => { unlisten = fn; });
        return () => unlisten?.();
    }, []);

    const download = async () => {
        setDownloading(true);
        setError(null);
        try {
            setStatus(await invoke<Qwen3Status>('qwen3_download'));
        } catch (e) {
            setError(String(e));
        } finally {
            setDownloading(false);
            setProgress(null);
        }
    };

    const present = status?.artifactsPresent ?? false;

    return (
        <div className="space-y-3 rounded-xl border p-4">
            <div className="flex items-center justify-between">
                <div>
                    <div className="font-medium">Qwen3-ASR (Chinese, 0.6B int8)</div>
                    <div className="text-xs text-gray-500">
                        Local transcription engine — recommended for Chinese meetings
                    </div>
                </div>
                {present ? (
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                ) : status ? (
                    <XCircle className="h-5 w-5 text-red-500" />
                ) : (
                    <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
                )}
            </div>

            {!present && status && (
                <button
                    className="flex items-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
                    onClick={() => void download()}
                    disabled={downloading}
                >
                    {downloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                    {downloading
                        ? progress
                            ? progress.pct >= 0
                                ? `Downloading ${progress.stage}… ${progress.pct}%`
                                : `Downloading ${progress.stage}…`
                            : 'Starting…'
                        : 'Download model artifacts (~2 GB)'}
                </button>
            )}

            {progress && progress.pct >= 0 && downloading && (
                <div className="h-2 w-full overflow-hidden rounded bg-gray-200 dark:bg-gray-700">
                    <div className="h-full bg-blue-600" style={{ width: `${progress.pct}%` }} />
                </div>
            )}

            {error && (
                <div className="rounded border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
                    {error}
                </div>
            )}

            {status && !present && status.missing.length > 0 && (
                <div className="text-xs text-gray-500">Missing: {status.missing.join(', ')}</div>
            )}

            {status && (
                <div className="text-xs text-gray-400">Models dir: {status.modelsDir}</div>
            )}
        </div>
    );
}

export function Qwen3Select({
    onSave,
    disabled,
}: {
    onSave: (provider: 'localWhisper' | 'parakeet' | 'qwen3', model: string) => Promise<boolean>;
    disabled?: boolean;
}) {
    const [present, setPresent] = useState<boolean | null>(null);
    const [selected, setSelected] = useState(false);

    useEffect(() => {
        void invoke<Qwen3Status>('qwen3_status').then((s) => setPresent(s.artifactsPresent));
    }, []);

    if (present === null || !present) return null;

    return (
        <button
            className={`w-full rounded-md border px-3 py-2 text-sm ${selected ? 'border-[var(--af-accent)] bg-[var(--af-accent)]/10' : 'hover:bg-gray-50 dark:hover:bg-gray-800'}`}
            disabled={disabled}
            onClick={async () => {
                const ok = await onSave('qwen3', 'qwen3-asr-0.6B-int8');
                if (ok) setSelected(true);
            }}
        >
            {selected ? '✓ Using Qwen3-ASR (Chinese)' : 'Use Qwen3-ASR for transcription (Chinese)'}
        </button>
    );
}
