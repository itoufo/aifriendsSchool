'use client';

import { useEffect, useState } from 'react';
import './ProgressBar.css';

interface ProgressBarProps {
  current: number;
  total: number;
  label?: string;
  showPercentage?: boolean;
  color?: 'primary' | 'success' | 'warning';
  size?: 'small' | 'medium' | 'large';
}

export const ProgressBar = ({ 
  current, 
  total, 
  label,
  showPercentage = true,
  color = 'primary',
  size = 'medium'
}: ProgressBarProps) => {
  const [percentage, setPercentage] = useState(0);

  useEffect(() => {
    const newPercentage = total > 0 ? Math.round((current / total) * 100) : 0;
    setTimeout(() => setPercentage(newPercentage), 100);
  }, [current, total]);

  return (
    <div className={`progress-bar-container ${size}`}>
      {label && <div className="progress-label">{label}</div>}
      <div className="progress-bar-wrapper">
        <div className="progress-bar-track">
          <div 
            className={`progress-bar-fill ${color}`}
            style={{ width: `${percentage}%` }}
          >
            {showPercentage && percentage > 10 && (
              <span className="progress-percentage-inside">{percentage}%</span>
            )}
          </div>
        </div>
        {showPercentage && percentage <= 10 && (
          <span className="progress-percentage-outside">{percentage}%</span>
        )}
      </div>
      <div className="progress-stats">
        <span>{current} / {total} 完了</span>
      </div>
    </div>
  );
};