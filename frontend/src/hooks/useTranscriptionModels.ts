import { useState, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { isVisibleParakeetModel } from '@/lib/parakeet';

export interface RawModelInfo {
  name: string;
  size_mb: number;
  status: 'Available' | 'Missing' | { Downloading: { progress: number } } | { Error: string };
}

export interface ModelOption {
  provider: 'whisper' | 'parakeet' | 'qwen3';
  name: string;
  displayName: string;
  size_mb: number;
}

interface TranscriptModelConfig {
  provider?: string;
  model?: string;
}

/**
 * Custom hook for fetching and managing transcription models (Whisper and Parakeet).
 *
 * This hook centralizes the model fetching logic that was previously duplicated
 * in ImportAudioDialog and RetranscribeDialog components.
 *
 * @param transcriptModelConfig - User's saved model configuration from context
 * @returns Object containing available models, selected model key, loading state, and fetch function
 */
export function useTranscriptionModels(transcriptModelConfig: TranscriptModelConfig | undefined) {
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [hasWhisperModel, setHasWhisperModel] = useState(false);
  const [hasParakeetModel, setHasParakeetModel] = useState(false);
  const [selectedModelKey, setSelectedModelKey] = useState<string>('');
  const [loadingModels, setLoadingModels] = useState(false);
  // Track whether the user has manually changed the model selection
  const userSelectedRef = useRef(false);

  // Wrap setSelectedModelKey to track user-initiated changes
  const setSelectedModelKeyWithTracking = useCallback((key: string) => {
    userSelectedRef.current = true;
    setSelectedModelKey(key);
  }, []);

  const fetchModels = useCallback(async (preferredConfig?: TranscriptModelConfig) => {
    setLoadingModels(true);
    const allModels: ModelOption[] = [];

    // Fetch Whisper models
    try {
      const whisperModels = await invoke<RawModelInfo[]>('whisper_get_available_models');
      const availableWhisper = whisperModels
        .filter((m) => m.status === 'Available')
        .map((m) => ({
          provider: 'whisper' as const,
          name: m.name,
          displayName: `🏠 Whisper: ${m.name}`,
          size_mb: m.size_mb,
        }));
      setHasWhisperModel(availableWhisper.length > 0);
      allModels.push(...availableWhisper);
    } catch (err) {
      console.error('Failed to fetch Whisper models:', err);
      setHasWhisperModel(false);
    }

    // Fetch Parakeet models
    try {
      const parakeetModels = await invoke<RawModelInfo[]>('parakeet_get_available_models');
      const availableParakeet = parakeetModels
        .filter((m) => m.status === 'Available' && isVisibleParakeetModel(m.name))
        .map((m) => ({
          provider: 'parakeet' as const,
          name: m.name,
          displayName: `⚡ Parakeet: ${m.name}`,
          size_mb: m.size_mb,
        }));
      setHasParakeetModel(availableParakeet.length > 0);
      allModels.push(...availableParakeet);
    } catch (err) {
      console.error('Failed to fetch Parakeet models:', err);
      setHasParakeetModel(false);
    }

    // Fetch Qwen3 availability (single model entry when artifacts present)
    try {
      const st = await invoke<{ artifactsPresent: boolean }>('qwen3_status');
      if (st.artifactsPresent) {
        allModels.push({
          provider: 'qwen3' as const,
          name: 'qwen3-asr-0.6B-int8',
          displayName: '🇨🇳 Qwen3-ASR (Chinese)',
          size_mb: 0,
        });
      }
    } catch (err) {
      console.error('Failed to fetch Qwen3 status:', err);
    }

    setAvailableModels(allModels);

    // Set default model based on user's saved configuration
    const effectiveConfig = preferredConfig || transcriptModelConfig;
    const configuredProvider = effectiveConfig?.provider || '';
    const configuredModel = effectiveConfig?.model || '';

    // Try to match the configured model
    // Note: 'localWhisper' in config maps to 'whisper' provider in model list
    const configuredMatch = allModels.find(
      (m) =>
        ((configuredProvider === 'localWhisper' || configuredProvider === 'whisper') && m.provider === 'whisper' && m.name === configuredModel) ||
        (configuredProvider === 'parakeet' && m.provider === 'parakeet' && m.name === configuredModel) ||
        (configuredProvider === 'qwen3' && m.provider === 'qwen3' && m.name === configuredModel)
    );
    const normalizedProvider = configuredProvider === 'localWhisper' ? 'whisper' : configuredProvider;
    const configuredProviderMatch = allModels.find((model) => model.provider === normalizedProvider);

    // Only set default model if user hasn't manually selected one
    if (!userSelectedRef.current) {
      if (configuredMatch) {
        // Use the configured model if available
        setSelectedModelKey(`${configuredMatch.provider}:${configuredMatch.name}`);
      } else if (configuredProviderMatch) {
        // Preserve the post-call provider when its exact model was removed.
        setSelectedModelKey(`${configuredProviderMatch.provider}:${configuredProviderMatch.name}`);
      } else if (allModels.length > 0) {
        // Fall back to first available model
        setSelectedModelKey(`${allModels[0].provider}:${allModels[0].name}`);
      }
    }

    setLoadingModels(false);
  }, [transcriptModelConfig]);

  // Reset user selection tracking (call when dialog opens fresh)
  const resetSelection = useCallback(() => {
    userSelectedRef.current = false;
  }, []);

  return {
    availableModels,
    selectedModelKey,
    setSelectedModelKey: setSelectedModelKeyWithTracking,
    loadingModels,
    hasWhisperModel,
    hasParakeetModel,
    fetchModels,
    resetSelection,
  };
}
