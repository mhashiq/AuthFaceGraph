/**
 * AuthBrain AI Face Analysis Engine
 * SystemLogs Component
 *
 * Scrollable, color-coded system event log panel.
 */

import React, { useRef, useEffect } from 'react';
import { Terminal, Trash2 } from 'lucide-react';
import { useAnalysisStore } from '../../store';
import clsx from 'clsx';

const LEVEL_CONFIG = {
  info:    { cls: 'text-dark-300', prefix: '[INFO]',  prefix_cls: 'text-metric-focus' },
  warning: { cls: 'text-risk-medium', prefix: '[WARN]', prefix_cls: 'text-risk-medium' },
  error:   { cls: 'text-risk-critical', prefix: '[ERR]', prefix_cls: 'text-risk-critical' },
};

export const SystemLogs: React.FC = () => {
  const logs         = useAnalysisStore(s => s.logs);
  const scrollRef    = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest log
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [logs.length]);

  return (
    <div className="bg-dark-900/60 border border-dark-600/50 rounded-xl flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-dark-700/50">
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-brand-500" />
          <span className="text-xs font-mono font-semibold text-white uppercase tracking-wider">
            System Logs
          </span>
          <span className="text-xs font-mono text-dark-500">({logs.length})</span>
        </div>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto max-h-48 p-3 space-y-0.5 font-mono"
        style={{ scrollbarWidth: 'thin' }}
      >
        {logs.length === 0 ? (
          <p className="text-dark-600 text-xs py-4 text-center">No log entries yet</p>
        ) : (
          logs.map((log) => {
            const cfg = LEVEL_CONFIG[log.level];
            const time = new Date(log.timestamp).toLocaleTimeString('en-US', {
              hour12: false,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            });
            return (
              <div key={log.id} className="flex gap-2 text-xs leading-relaxed">
                <span className="text-dark-600 flex-shrink-0 w-18">{time}</span>
                <span className={clsx('flex-shrink-0 w-12', cfg.prefix_cls)}>{cfg.prefix}</span>
                <span className="text-dark-500 flex-shrink-0">[{log.source}]</span>
                <span className={cfg.cls}>{log.message}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
