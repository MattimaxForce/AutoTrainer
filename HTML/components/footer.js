class CustomFooter extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        footer {
          background: rgba(17, 24, 39, 0.8);
          backdrop-filter: blur(10px);
          color: rgba(255, 255, 255, 0.7);
          padding: 2rem;
          text-align: center;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          margin-top: auto;
        }
        .footer-content {
          max-width: 1200px;
          margin: 0 auto;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
        }
        .footer-links {
          display: flex;
          gap: 1.5rem;
          margin-bottom: 1rem;
        }
        .footer-links a {
          color: rgba(255, 255, 255, 0.7);
          text-decoration: none;
          transition: color 0.2s;
        }
        .footer-links a:hover {
          color: white;
        }
        .copyright {
          font-size: 0.875rem;
        }
        .social-icons {
          display: flex;
          gap: 1rem;
          margin-top: 1rem;
        }
        .social-icons a {
          color: rgba(255, 255, 255, 0.7);
          transition: color 0.2s;
        }
        .social-icons a:hover {
          color: white;
        }
      </style>
      <footer>
        <div class="footer-content">
          <div class="footer-links">
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Service</a>
            <a href="#">Documentation</a>
            <a href="#">Contact</a>
          </div>
          <div class="social-icons">
            <a href="#"><i data-feather="github"></i></a>
            <a href="#"><i data-feather="twitter"></i></a>
            <a href="#"><i data-feather="linkedin"></i></a>
          </div>
          <p class="copyright">&copy; ${new Date().getFullYear()} HF AutoTrain Wizard. All rights reserved.</p>
        </div>
      </footer>
    `;
  }
}
customElements.define('custom-footer', CustomFooter);