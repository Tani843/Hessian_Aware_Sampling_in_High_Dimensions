# Hessian Aware Sampling Documentation Website

This is a complete Jekyll documentation website for the Hessian Aware Sampling in High Dimensions research project. The site provides comprehensive documentation of the theoretical foundations, methodology, results, and practical applications of Hessian-aware MCMC methods.

## Site Structure

```
docs/jekyll_site/
├── _config.yml              # Jekyll configuration
├── index.md                 # Home page
├── _layouts/                # Page layouts
│   ├── default.html         # Main layout template
│   └── page.html           # Content page layout
├── _includes/              # Reusable components
│   ├── header.html         # Site header
│   ├── footer.html         # Site footer
│   └── navigation.html     # Navigation menu
├── pages/                  # Content pages
│   ├── about.md           # Problem motivation and theory
│   ├── methodology.md     # Detailed algorithms
│   ├── results.md         # Benchmarking and analysis
│   ├── conclusion.md      # Summary and future work
│   └── contact.md         # Author info and resources
├── assets/                # Static assets
│   ├── css/main.css       # Custom styling
│   ├── js/main.js         # Interactive functionality
│   └── images/            # Plots and diagrams
├── _data/                 # Site data
│   └── navigation.yml     # Navigation configuration
└── Gemfile               # Ruby dependencies
```

## Features

- **Clean, Minimal Design**: Black text on white background with professional typography
- **Mathematical Content**: MathJax integration for LaTeX equations
- **Interactive Elements**: Copy buttons for code blocks, image modals, smooth scrolling
- **Responsive Design**: Mobile-friendly layout with adaptive navigation
- **Table of Contents**: Auto-generated TOC for long pages
- **Syntax Highlighting**: Rouge syntax highlighter for code blocks
- **Professional Styling**: Publication-quality presentation

## Local Development

### Prerequisites

- Ruby 2.7 or higher
- Bundler gem

### Setup

1. Install dependencies:
   ```bash
   cd docs/jekyll_site
   bundle install
   ```

2. Start development server:
   ```bash
   bundle exec jekyll serve
   ```

3. View the site at `http://localhost:4000`

### Development Commands

```bash
# Build the site
bundle exec jekyll build

# Serve with live reload
bundle exec jekyll serve --livereload

# Serve on specific port
bundle exec jekyll serve --port 4001

# Build for production
JEKYLL_ENV=production bundle exec jekyll build
```

## Deployment

### GitHub Pages

1. Push to GitHub repository
2. Enable GitHub Pages in repository settings
3. Select source branch (usually `main` or `gh-pages`)
4. Site will be available at `https://username.github.io/repository-name`

### Other Hosting

The site generates static HTML/CSS/JS files in the `_site` directory that can be deployed to any web server:

- Netlify
- Vercel
- AWS S3
- Apache/Nginx

## Customization

### Styling

The main stylesheet is `assets/css/main.css`. Key design elements:

- **Typography**: System fonts with serif fallbacks
- **Colors**: Minimal palette with black text on white background  
- **Layout**: Clean grid system with proper spacing
- **Components**: Styled boxes for algorithms, results, and highlights

### Content

All content is in Markdown format in the `pages/` directory. Each page includes:

- Front matter with title, subtitle, and permalink
- Structured content with headings and sections
- Mathematical equations using LaTeX syntax
- Code blocks with syntax highlighting
- Custom styled boxes for important content

### Navigation

Navigation is controlled by `_data/navigation.yml`. To add new pages:

1. Create the Markdown file in `pages/`
2. Add front matter with title and permalink
3. Add entry to navigation.yml

## Content Guidelines

### Mathematical Notation

Use LaTeX syntax for mathematics:
- Inline: `$equation$` 
- Display: `$$equation$$`
- Numbered equations: Use `\begin{equation}...\end{equation}`

### Code Blocks

Use fenced code blocks with language specification:
```python
def example_function():
    return "Hello World"
```

### Custom Styling

Use custom CSS classes for special content:
- `.highlight-box`: Important information
- `.algorithm-box`: Algorithm descriptions  
- `.result-box`: Key results and findings
- `.quick-start`: Getting started sections

### Images and Figures

Images are stored in `assets/images/`. Reference them using:
```markdown
![Alt text]({{ '/assets/images/filename.png' | relative_url }})
```

## Browser Compatibility

The site is tested and compatible with:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

The site is optimized for performance:
- Minimal JavaScript (< 5KB)
- Optimized CSS (< 50KB)
- Lazy loading for images
- Efficient MathJax configuration

## Accessibility

The site follows accessibility best practices:
- Semantic HTML structure
- ARIA labels where appropriate
- Keyboard navigation support
- High contrast text
- Alternative text for images

## License

The documentation and code are released under the MIT License. See the main project repository for full license details.

## Contributing

Contributions to improve the documentation are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## Support

For questions about the documentation site:
- Open an issue in the main repository
- Contact the authors through the information provided on the Contact page
- Check the Jekyll documentation for general Jekyll questions