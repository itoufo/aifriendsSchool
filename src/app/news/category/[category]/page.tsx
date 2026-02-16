import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { BASE_SEO_KEYWORDS } from '@/config/seo';
import { absoluteUrl, SITE } from '@/config/site';
import {
  formatNewsDate,
  getAllNewsCategories,
  getCategoryNameBySlug,
  getNewsByCategorySlug,
} from '@/data/news';
import styles from '../../news.module.css';

interface PageProps {
  params: Promise<{ category: string }>;
}

export async function generateStaticParams() {
  return getAllNewsCategories().map((category) => ({
    category: category.slug,
  }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { category } = await params;
  const categoryName = getCategoryNameBySlug(category);
  if (!categoryName) {
    return {
      robots: {
        index: false,
        follow: false,
      },
    };
  }

  const canonicalPath = `/news/category/${category}`;
  const description = `${categoryName}に関するAIニュースと実務解説をまとめて確認できます。`;

  return {
    title: `${categoryName}のAIニュース`,
    description,
    keywords: [...BASE_SEO_KEYWORDS, 'AIニュース', categoryName],
    alternates: {
      canonical: canonicalPath,
    },
    openGraph: {
      type: 'website',
      title: `${categoryName} | AIニュース`,
      description,
      url: canonicalPath,
      locale: SITE.locale,
      siteName: SITE.name,
      images: [
        {
          url: '/news/opengraph-image',
          width: 1200,
          height: 630,
          alt: `${categoryName} AIニュース`,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title: `${categoryName} | AIニュース`,
      description,
      images: ['/news/opengraph-image'],
    },
  };
}

export default async function NewsCategoryPage({ params }: PageProps) {
  const { category } = await params;
  const categoryName = getCategoryNameBySlug(category);
  if (!categoryName) {
    notFound();
  }

  const articles = getNewsByCategorySlug(category);
  if (articles.length === 0) {
    notFound();
  }

  const categories = getAllNewsCategories();
  const canonicalUrl = absoluteUrl(`/news/category/${category}`);

  const jsonLd = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'CollectionPage',
        '@id': `${canonicalUrl}#collection`,
        url: canonicalUrl,
        name: `${categoryName}のAIニュース`,
        description: `${categoryName}に関連するニュース・実務解説記事一覧。`,
        inLanguage: SITE.language,
      },
      {
        '@type': 'ItemList',
        '@id': `${canonicalUrl}#item-list`,
        name: `${categoryName}カテゴリ記事`,
        numberOfItems: articles.length,
        itemListElement: articles.map((article, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          name: article.title,
          url: absoluteUrl(`/news/${article.slug}`),
        })),
      },
    ],
  };

  return (
    <div className={styles.page}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <div className={styles.container}>
        <nav className={styles.breadcrumb} aria-label="パンくずリスト">
          <Link href="/">ホーム</Link>
          <span>/</span>
          <Link href="/news">AIニュース</Link>
          <span>/</span>
          <span>{categoryName}</span>
        </nav>

        <section className={styles.hero}>
          <h1 className={styles.title}>{categoryName}のAIニュース</h1>
          <p className={styles.description}>
            {categoryName}に関連する最新記事をまとめています。実務で使える観点に絞って解説しています。
          </p>
          <div className={styles.categoryLinks}>
            {categories.map((item) => (
              <Link
                key={item.slug}
                href={`/news/category/${item.slug}`}
                className={styles.categoryLink}
              >
                {item.name}
                <span>{item.count}</span>
              </Link>
            ))}
          </div>
        </section>

        <section className={styles.list}>
          {articles.map((article) => (
            <article key={article.slug} className={styles.card}>
              <div className={styles.metaRow}>
                <Link
                  href={`/news/category/${article.categorySlug}`}
                  className={styles.category}
                >
                  {article.category}
                </Link>
                <time className={styles.date} dateTime={article.publishedAt}>
                  {formatNewsDate(article.publishedAt)}
                  {article.updatedAt ? ` 更新: ${formatNewsDate(article.updatedAt)}` : ''}
                </time>
              </div>
              <h2 className={styles.cardTitle}>
                <Link href={`/news/${article.slug}`}>{article.title}</Link>
              </h2>
              <p className={styles.summary}>{article.summary}</p>
              <div className={styles.tagList}>
                {article.tags.map((tag) => (
                  <span key={tag} className={styles.tag}>
                    #{tag}
                  </span>
                ))}
              </div>
              <Link href={`/news/${article.slug}`} className={styles.readMore}>
                記事を読む
              </Link>
            </article>
          ))}
        </section>
      </div>
    </div>
  );
}
