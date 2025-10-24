class CustomNavbar extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        nav {
          background: rgba(17, 24, 39, 0.8);
          backdrop-filter: blur(10px);
          padding: 1rem 2rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .logo-container {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        .logo {
          font-size: 1.25rem;
          font-weight: 600;
          background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .hf-logo {
          height: 1.75rem;
        }
        ul {
          display: flex;
          gap: 1.5rem;
          list-style: none;
          margin: 0;
          padding: 0;
        }
        a {
          color: rgba(255, 255, 255, 0.8);
          text-decoration: none;
          font-weight: 500;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        a:hover {
          color: white;
        }
        .nav-icon {
          width: 1.25rem;
          height: 1.25rem;
        }
        @media (max-width: 768px) {
          nav {
            padding: 1rem;
          }
          ul {
            gap: 1rem;
          }
        }
      </style>
      <nav>
        <div class="logo-container">
          <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" class="hf-logo">
          <span class="logo">AutoTrain</span>
        </div>
        <ul>
          <li><a href="#"><i data-feather="home" class="nav-icon"></i> Home</a></li>
          <li><a href="#"><i data-feather="book" class="nav-icon"></i> Docs</a></li>
          <li><a href="#"><i data-feather="settings" class="nav-icon"></i> Settings</a></li>
        </ul>
      </nav>
    `;
  }
}
customElements.define('custom-navbar', CustomNavbar);