import { ImageResponse } from 'next/og';
import { SITE } from '@/config/site';
import { getDocById, getSectionByDocId } from '@/data/curriculum';

export const runtime = 'edge';

export const alt = 'AI Friends School Lesson';
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = 'image/png';

export default function OpenGraphImage({
  params,
}: {
  params: { docId: string };
}) {
  const doc = getDocById(params.docId);
  const section = getSectionByDocId(params.docId);

  const title = doc?.title ?? '教材';
  const subtitle = section?.title ?? SITE.name;

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
            'linear-gradient(135deg, #0b1220 0%, #0b1220 35%, #38bdf8 145%)',
          padding: 64,
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: 1000,
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
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
              justifyContent: 'space-between',
              gap: 18,
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 14,
                color: 'rgba(255,255,255,0.85)',
                fontSize: 26,
              }}
            >
              <div
                style={{
                  width: 46,
                  height: 46,
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
                color: 'rgba(255,255,255,0.72)',
                fontSize: 22,
              }}
            >
              /doc/{params.docId}
            </div>
          </div>

          <div
            style={{
              color: 'rgba(255,255,255,0.88)',
              fontSize: 28,
              fontWeight: 700,
              letterSpacing: 0.2,
            }}
          >
            {subtitle}
          </div>

          <div
            style={{
              color: '#fff',
              fontSize: 58,
              fontWeight: 900,
              lineHeight: 1.12,
              wordBreak: 'break-word',
            }}
          >
            {title}
          </div>

          <div
            style={{
              marginTop: 6,
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

