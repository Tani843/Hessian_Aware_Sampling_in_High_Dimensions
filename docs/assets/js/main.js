// Main JavaScript functionality for the Hessian Sampling documentation site

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add copy button to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', function() {
            navigator.clipboard.writeText(block.textContent).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        const pre = block.parentElement;
        pre.style.position = 'relative';
        pre.appendChild(button);
    });

    // Highlight current section in navigation on scroll
    const sections = document.querySelectorAll('h2[id], h3[id]');
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    
    function highlightCurrentSection() {
        let currentSection = '';
        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 100) {
                currentSection = section.id;
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('current-section');
            if (link.getAttribute('href') === '#' + currentSection) {
                link.classList.add('current-section');
            }
        });
    }

    window.addEventListener('scroll', highlightCurrentSection);

    // Mobile menu toggle (if needed in future)
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const navigation = document.querySelector('.site-navigation');
    
    if (mobileMenuToggle && navigation) {
        mobileMenuToggle.addEventListener('click', function() {
            navigation.classList.toggle('mobile-open');
        });
    }

    // Image lazy loading and lightbox functionality
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('click', function() {
            if (this.src.includes('plot') || this.src.includes('diagram')) {
                openImageModal(this.src, this.alt);
            }
        });
    });

    function openImageModal(src, alt) {
        const modal = document.createElement('div');
        modal.className = 'image-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <img src="${src}" alt="${alt}">
                <p class="modal-caption">${alt}</p>
            </div>
        `;
        
        document.body.appendChild(modal);
        modal.style.display = 'flex';
        
        const closeBtn = modal.querySelector('.close-modal');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }

    // Table of contents generator for long pages
    function generateTableOfContents() {
        const headings = document.querySelectorAll('h2, h3');
        if (headings.length > 3) {
            const toc = document.createElement('div');
            toc.className = 'table-of-contents';
            toc.innerHTML = '<h3>Table of Contents</h3><ul></ul>';
            
            const list = toc.querySelector('ul');
            headings.forEach(heading => {
                if (!heading.id) {
                    heading.id = heading.textContent.toLowerCase()
                        .replace(/[^\w\s-]/g, '')
                        .replace(/\s+/g, '-');
                }
                
                const listItem = document.createElement('li');
                listItem.innerHTML = `<a href="#${heading.id}">${heading.textContent}</a>`;
                listItem.className = heading.tagName.toLowerCase();
                list.appendChild(listItem);
            });
            
            const content = document.querySelector('.page-content');
            if (content && content.children.length > 0) {
                content.insertBefore(toc, content.children[0]);
            }
        }
    }

    // Generate TOC for methodology and results pages
    if (window.location.pathname.includes('methodology') || 
        window.location.pathname.includes('results')) {
        generateTableOfContents();
    }

    console.log('Hessian Sampling documentation site loaded successfully');
});