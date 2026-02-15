import type { Metadata } from 'next';
import { HomePage } from '@/components/HomePage';
import { absoluteUrl, SITE } from '@/config/site';

export const metadata: Metadata = {
  title: SITE.title,
  description: SITE.description,
  alternates: {
    canonical: '/',
  },
  openGraph: {
    url: '/',
  },
};

export default function Page() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'EducationalOrganization',
            '@id': `${SITE.url}/#org`,
            name: SITE.name,
            description: SITE.description,
            url: SITE.url,
            logo: absoluteUrl('/images/logo.png'),
            inLanguage: SITE.language,
          }),
        }}
      />
      <HomePage />
    </>
  );
}
