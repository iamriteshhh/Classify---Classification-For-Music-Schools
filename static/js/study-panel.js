/**
 * study-panel.js
 * ==============
 * Progressive enhancement for the Genre Reference Library two-pane app shell.
 * Intercepts sidebar clicks, fetches detail fragments via AJAX, swaps content
 * in place without full page reloads, and synchronizes browser history/back/forward.
 * Part of Final Design Specification (final_design.md §3).
 */

(function () {
    'use strict';

    window.StudyPanel = {
        init: function () {
            this.bindEvents();
            this.syncInitialState();
        },

        bindEvents: function () {
            const self = this;

            // Intercept clicks on sidebar genre links
            document.addEventListener('click', function (e) {
                const genreLink = e.target.closest('.app-shell__genre-item');
                if (genreLink && genreLink.getAttribute('data-ajax') !== 'false') {
                    e.preventDefault();
                    const url = genreLink.getAttribute('href');
                    const genreSlug = genreLink.getAttribute('data-slug');
                    self.loadGenre(url, genreSlug, true);
                }
            });

            // Handle browser back and forward
            window.addEventListener('popstate', function (e) {
                if (e.state && e.state.url) {
                    self.loadGenre(e.state.url, e.state.slug, false);
                } else {
                    // Back to base /library
                    self.showEmptyState();
                }
            });

            // Handle in-panel subnav pills smooth scrolling
            document.addEventListener('click', function (e) {
                const subnavPill = e.target.closest('.study-panel__subnav-pill');
                if (subnavPill) {
                    const targetId = subnavPill.getAttribute('href');
                    if (targetId && targetId.startsWith('#')) {
                        e.preventDefault();
                        const targetEl = document.querySelector(targetId);
                        if (targetEl) {
                            targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            // Update active pill
                            document.querySelectorAll('.study-panel__subnav-pill').forEach(p => p.classList.remove('active'));
                            subnavPill.classList.add('active');
                        }
                    }
                }
            });
        },

        syncInitialState: function () {
            const activeItem = document.querySelector('.app-shell__genre-item.active');
            if (activeItem) {
                const slug = activeItem.getAttribute('data-slug');
                const url = activeItem.getAttribute('href');
                if (window.history && window.history.replaceState) {
                    window.history.replaceState({ url: url, slug: slug }, '', url);
                }
                // If on mobile and a genre is pre-loaded, switch to detail view
                if (window.innerWidth < 1024) {
                    const shell = document.querySelector('.app-shell');
                    if (shell && document.querySelector('.study-panel')) {
                        shell.classList.add('mobile-view-detail');
                    }
                }
            }
        },

        loadGenre: function (url, slug, pushState) {
            const mainPanel = document.querySelector('.app-shell__main');
            const shell = document.querySelector('.app-shell');
            if (!mainPanel) return;

            // Highlight active item in sidebar
            document.querySelectorAll('.app-shell__genre-item').forEach(item => {
                if (item.getAttribute('data-slug') === slug) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });

            // Mobile viewport handling
            if (shell && window.innerWidth < 1024) {
                shell.classList.add('mobile-view-detail');
            }

            // Fetch fragment via AJAX
            fetch(url, {
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Accept': 'text/html',
                }
            })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error('Network response was not ok: ' + response.status);
                }
                return response.text();
            })
            .then(function (html) {
                mainPanel.innerHTML = html;
                mainPanel.scrollTop = 0;

                if (pushState && window.history && window.history.pushState) {
                    window.history.pushState({ url: url, slug: slug }, '', url);
                }
            })
            .catch(function (err) {
                console.error('Failed to load genre fragment:', err);
                // Graceful fallback to browser full page load on network error
                window.location.href = url;
            });
        },

        showEmptyState: function () {
            const mainPanel = document.querySelector('.app-shell__main');
            const shell = document.querySelector('.app-shell');
            if (!mainPanel) return;

            document.querySelectorAll('.app-shell__genre-item').forEach(item => item.classList.remove('active'));

            if (shell) {
                shell.classList.remove('mobile-view-detail');
            }

            mainPanel.innerHTML = `
                <div class="study-panel__empty">
                    <div class="mono" style="font-size: 2rem; color: var(--text-muted);">&#x266B;</div>
                    <h2 class="study-panel__empty-title">Select a Genre to Study</h2>
                    <p class="study-panel__empty-desc">
                        Choose any genre from the catalog to explore its acoustic profile,
                        historical lineage, core instrumentation, and pedagogical teaching concepts.
                    </p>
                </div>
            `;
        },

        closeMobileDetail: function () {
            const shell = document.querySelector('.app-shell');
            if (shell) {
                shell.classList.remove('mobile-view-detail');
            }
        }
    };

    // Auto-initialize when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            StudyPanel.init();
        });
    } else {
        StudyPanel.init();
    }
})();
