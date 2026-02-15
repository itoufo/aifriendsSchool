'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Sidebar } from './Sidebar';
import { AppConfig } from '../config/app.config';
import './Layout.css';

export const LayoutShell = ({ children }: { children: React.ReactNode }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="layout">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="main-area">
        <header className="top-bar">
          <button
            className="menu-btn"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12h18M3 6h18M3 18h18" />
            </svg>
          </button>
          <Link href="/" className="top-bar-title" aria-label="ホームへ">
            {AppConfig.branding.title}
          </Link>
          <img
            src={AppConfig.branding.logo.src}
            alt={AppConfig.branding.logo.alt}
            className="top-bar-logo"
          />
        </header>
        <main className="main-content">
          {children}
        </main>
        <footer className="site-footer">
          <p>{AppConfig.branding.footer.text}</p>
        </footer>
      </div>
    </div>
  );
};
