import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { HomePage } from './pages/HomePage';
import { DocPage } from './pages/DocPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="doc/:docId" element={<DocPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
