import { ImageResponse } from 'next/og';
import { SITE } from '@/config/site';

export const runtime = 'edge';

export const alt = SITE.title;
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = 'image/png';

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background:
            'linear-gradient(135deg, #0b1220 0%, #0b1220 30%, #0ea5e9 130%)',
          padding: 64,
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
            width: '100%',
            maxWidth: 980,
            padding: 56,
            borderRadius: 32,
            background:
              'linear-gradient(135deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.06) 100%)',
            border: '1px solid rgba(255,255,255,0.14)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 14,
              color: 'rgba(255,255,255,0.85)',
              fontSize: 28,
              letterSpacing: 1,
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 14,
                background:
                  'linear-gradient(135deg, rgba(14,165,233,1) 0%, rgba(99,102,241,1) 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                fontWeight: 800,
                fontSize: 22,
              }}
            >
              AI
            </div>
            <div style={{ fontWeight: 700 }}>{SITE.name}</div>
          </div>

          <div
            style={{
              color: '#fff',
              fontSize: 64,
              fontWeight: 900,
              lineHeight: 1.1,
            }}
          >
            Structured AI Curriculum
          </div>

          <div
            style={{
              color: 'rgba(255,255,255,0.92)',
              fontSize: 32,
              lineHeight: 1.4,
            }}
          >
            Beginner to Executive. 5 levels, practical and systematic.
          </div>

          <div
            style={{
              marginTop: 8,
              color: 'rgba(255,255,255,0.70)',
              fontSize: 22,
            }}
          >
            {SITE.url.replace(/^https?:\/\//, '')}
          </div>
        </div>
      </div>
    ),
    size
  );
}

