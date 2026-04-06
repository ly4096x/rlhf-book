class NavigationDropdown extends HTMLElement {
    constructor() {
      super();

      // Get the initial expanded state from the attribute, default to false
      const initialExpanded = this.getAttribute('expanded') === 'true';

      this.innerHTML = `
        <div>
          <button class="dropdown-button" aria-expanded="${initialExpanded}">
            <span><strong>Navigation</strong></span>
            <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>

          <div class="dropdown-content${initialExpanded ? ' open' : ''}">
    <nav class="chapter-nav">
      <div class="section">
        <h3>链接</h3>
        <ul>
          <li><a href="../../">主页</a> / <a href="https://github.com/ly4096x/rlhf-book">GitHub</a></li>
          <li><a href="../../book.pdf">PDF</a> / <a href="https://arxiv.org/abs/2504.12501">Arxiv</a> / <a href="../../book.epub">EPUB</a> / <a href="../../book.kindle.epub">Kindle</a></li>
          <li><a href="https://www.manning.com/books/the-rlhf-book">预购</a></li>
        </ul>
        <h3>资源</h3>
        <ul>
          <li><a href="../../rl-cheatsheet">RL 速查表</a></li>
          <li><a href="../../library">补全库</a></li>
          <li><a href="../../course">课程</a></li>
        </ul>
      </div>

      <div class="section">
        <h3>引言</h3>
        <ol start="1">
          <li><a href="01-introduction">引言</a></li>
          <li><a href="02-related-works">关键相关工作</a></li>
          <li><a href="03-training-overview">训练概述</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>核心训练流程</h3>
        <ol start="4">
          <li><a href="04-instruction-tuning">指令微调</a></li>
          <li><a href="05-reward-models">奖励模型</a></li>
          <li><a href="06-policy-gradients">强化学习</a></li>
          <li><a href="07-reasoning">推理</a></li>
          <li><a href="08-direct-alignment">直接对齐</a></li>
          <li><a href="09-rejection-sampling">拒绝采样</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>数据与偏好</h3>
        <ol start="10">
          <li><a href="10-preferences">什么是偏好</a></li>
          <li><a href="11-preference-data">偏好数据</a></li>
          <li><a href="12-synthetic-data">合成数据与 CAI</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>实践考量</h3>
        <ol start="13">
          <li><a href="13-tools">工具使用</a></li>
          <li><a href="14-over-optimization">过度优化</a></li>
          <li><a href="15-regularization">正则化</a></li>
          <li><a href="16-evaluation">评估</a></li>
          <li><a href="17-product">产品与角色</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>附录</h3>
        <ol type="A" style="padding-left: 0; list-style-position: inside;">
          <li><a href="appendix-a-definitions">定义</a></li>
          <li><a href="appendix-b-style">风格与信息</a></li>
          <li><a href="appendix-c-practical">实践问题</a></li>
        </ol>
      </div>
    </nav>
    <div id="search"></div>
  </div>
</div>
      `;

      // Initialize Pagefind search if available
      var searchEl = this.querySelector('#search');
      if (searchEl && typeof PagefindUI !== 'undefined') {
        new PagefindUI({ element: searchEl, showImages: false });
      }

      // Set up click handler
      const button = this.querySelector('.dropdown-button');
      const content = this.querySelector('.dropdown-content');

      button.addEventListener('click', () => {
        const isExpanded = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', !isExpanded);
        content.classList.toggle('open');
      });
    }

    // Add attribute change observer
    static get observedAttributes() {
      return ['expanded'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
      if (name === 'expanded') {
        const button = this.querySelector('.dropdown-button');
        const content = this.querySelector('.dropdown-content');
        const isExpanded = newValue === 'true';

        if (button && content) {
          button.setAttribute('aria-expanded', isExpanded);
          content.classList.toggle('open', isExpanded);
        }
      }
    }
}

// Only define the component once
if (!customElements.get('navigation-dropdown')) {
  customElements.define('navigation-dropdown', NavigationDropdown);
}
