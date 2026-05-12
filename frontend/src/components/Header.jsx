import '../styles/Header.scss';

export default function Header() {
  return (
    <header className="header">
      <div className="container header-inner">
        <div className="header-brand">
          <div>
            <h1 className="header-title">
              <span className="gradient-text">Gender Voice</span> AI
            </h1>
            <p className="header-subtitle">Deep Learning Classification</p>
          </div>
        </div>
        <div className="header-badge">
          <span className="badge-dot"></span>
          TCN + Transformer + Attention
        </div>
      </div>
    </header>
  );
}
