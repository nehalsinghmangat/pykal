/* JavaScript for bibliography filtering with implementation tracking */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Bibliography script loaded');

    // Get all filter elements
    const categoryFilter = document.getElementById('category-filter');
    const implCheckboxes = document.querySelectorAll('input.impl-checkbox');
    const resetButton = document.getElementById('reset-filters');
    const noResultsMessage = document.getElementById('no-results-message');

    // Get all bibliography entries - they're in an <ol> list
    const bibList = document.querySelector('#id1 ol.arabic');
    const citations = bibList ? bibList.querySelectorAll('li') : [];

    console.log(`Found ${citations.length} citations`);

    // PAPER_METADATA is loaded from bib_metadata.js
    const paperMetadata = typeof PAPER_METADATA !== 'undefined' ? PAPER_METADATA : {};
    console.log('Paper metadata:', paperMetadata);

    // Initialize citations with metadata and circles
    function initializeCitations() {
        citations.forEach((citation, index) => {
            const text = citation.textContent || citation.innerText;
            console.log(`Processing citation ${index}:`, text.substring(0, 100));

            // Try to find matching metadata
            let metadata = null;
            let matchedKey = null;

            for (const [key, data] of Object.entries(paperMetadata)) {
                // Extract author name from key (everything before first digit)
                const authorMatch = key.match(/^([a-z]+)/i);
                const authorName = authorMatch ? authorMatch[1] : '';

                // Extract year from citation text
                const yearMatch = text.match(/\b(19|20)\d{2}\b/);
                const year = yearMatch ? yearMatch[0] : '';

                // Match if both author name and year are found
                if (authorName && year &&
                    text.toLowerCase().includes(authorName.toLowerCase()) &&
                    key.toLowerCase().includes(year)) {
                    metadata = data;
                    matchedKey = key;
                    console.log(`Matched: "${authorName}" and "${year}" -> key: ${key}`);
                    break;
                }
            }

            if (metadata) {
                console.log(`Matched citation ${index} to key ${matchedKey}`);

                // Store metadata in dataset
                citation.dataset.keywords = metadata.keywords || '';
                citation.dataset.implPykal = metadata.impl_pykal || '';
                citation.dataset.implTurtlebot = metadata.impl_turtlebot || '';
                citation.dataset.implCrazyflie = metadata.impl_crazyflie || '';

                // Create circles container
                const circlesContainer = document.createElement('span');
                circlesContainer.className = 'citation-impl-badges';
                circlesContainer.style.display = 'inline-flex';
                circlesContainer.style.gap = '6px';
                circlesContainer.style.marginLeft = '10px';
                circlesContainer.style.verticalAlign = 'middle';

                // Add circles for each implementation
                const implementations = [
                    { key: 'impl_pykal', class: 'impl-pykal', label: 'pykal core', url: metadata.impl_pykal },
                    { key: 'impl_turtlebot', class: 'impl-turtlebot', label: 'TurtleBot', url: metadata.impl_turtlebot },
                    { key: 'impl_crazyflie', class: 'impl-crazyflie', label: 'Crazyflie', url: metadata.impl_crazyflie }
                ];

                let hasAnyImpl = false;
                implementations.forEach(impl => {
                    if (impl.url && impl.url.trim()) {
                        hasAnyImpl = true;
                        const circle = document.createElement('span');
                        circle.className = `impl-circle ${impl.class} clickable`;
                        circle.title = `Click to view ${impl.label} implementation`;
                        circle.style.display = 'inline-block';
                        circle.style.width = '14px';
                        circle.style.height = '14px';
                        circle.style.borderRadius = '50%';
                        circle.style.cursor = 'pointer';
                        circle.style.transition = 'transform 0.2s ease, box-shadow 0.2s ease';
                        circle.style.border = '2px solid rgba(0,0,0,0.1)';

                        // Set background color
                        if (impl.class === 'impl-pykal') {
                            circle.style.backgroundColor = '#007bff';
                        } else if (impl.class === 'impl-turtlebot') {
                            circle.style.backgroundColor = '#28a745';
                        } else if (impl.class === 'impl-crazyflie') {
                            circle.style.backgroundColor = '#ffc107';
                        }

                        circle.setAttribute('data-notebook-url', impl.url);
                        circle.addEventListener('click', function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            const notebookUrl = this.getAttribute('data-notebook-url');
                            console.log('Circle clicked, navigating to:', notebookUrl);
                            if (notebookUrl) {
                                // Convert .ipynb to .html for Sphinx build
                                const htmlUrl = notebookUrl.replace('.ipynb', '.html');
                                window.location.href = htmlUrl;
                            }
                        });

                        // Add hover effect
                        circle.addEventListener('mouseenter', function() {
                            this.style.transform = 'scale(1.2)';
                            this.style.boxShadow = '0 2px 6px rgba(0,0,0,0.3)';
                        });
                        circle.addEventListener('mouseleave', function() {
                            this.style.transform = 'scale(1)';
                            this.style.boxShadow = 'none';
                        });

                        circlesContainer.appendChild(circle);
                        console.log(`Added ${impl.class} circle for ${impl.label}`);

                        // Add border class
                        citation.classList.add(`has-${impl.class}`);
                    }
                });

                // Only insert circles container if there are implementations
                if (hasAnyImpl) {
                    // Insert circles at the beginning of the citation paragraph
                    const firstP = citation.querySelector('p');
                    if (firstP) {
                        firstP.insertBefore(circlesContainer, firstP.firstChild);
                        console.log('Circles inserted into citation');
                    } else {
                        // If no <p>, insert at beginning of <li>
                        citation.insertBefore(circlesContainer, citation.firstChild);
                        console.log('Circles inserted at start of <li>');
                    }
                } else {
                    console.log(`Citation ${index} has no implementations - no circles added`);
                }
            } else {
                console.log(`No metadata match for citation ${index}`);
            }
        });

        console.log('Citations initialized');
    }

    // Filter function
    function applyFilters() {
        console.log('Applying filters...');

        if (!categoryFilter || citations.length === 0) {
            console.log('No filters or citations found');
            return;
        }

        const categoryValue = categoryFilter.value;

        // Get selected implementation filters (can be multiple)
        const implValues = [];
        implCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                implValues.push(checkbox.value);
            }
        });

        console.log('Filter values:', { categoryValue, implValues });

        let visibleCount = 0;

        citations.forEach((citation, index) => {
            let visible = true;

            // Check category filter
            if (categoryValue !== 'all') {
                const keywords = citation.dataset.keywords || '';
                if (!keywords.includes(categoryValue)) {
                    visible = false;
                    console.log(`Citation ${index} hidden by category filter`);
                }
            }

            // Check implementation filter (only if checkboxes are selected)
            if (implValues.length > 0) {
                const implPykal = citation.dataset.implPykal || '';
                const implTurtlebot = citation.dataset.implTurtlebot || '';
                const implCrazyflie = citation.dataset.implCrazyflie || '';

                // Show paper if it has ANY of the selected implementations (OR logic)
                let hasSelectedImpl = false;
                if (implValues.includes('pykal') && implPykal) {
                    hasSelectedImpl = true;
                }
                if (implValues.includes('turtlebot') && implTurtlebot) {
                    hasSelectedImpl = true;
                }
                if (implValues.includes('crazyflie') && implCrazyflie) {
                    hasSelectedImpl = true;
                }

                if (!hasSelectedImpl) {
                    visible = false;
                    console.log(`Citation ${index} hidden - doesn't match selected implementations`);
                }
            }

            // Apply visibility
            if (visible) {
                citation.style.display = '';
                visibleCount++;
                console.log(`Citation ${index} visible`);
            } else {
                citation.style.display = 'none';
            }
        });

        console.log(`Visible citations: ${visibleCount}/${citations.length}`);

        // Show/hide no results message
        if (noResultsMessage) {
            if (visibleCount === 0) {
                noResultsMessage.style.display = 'block';
            } else {
                noResultsMessage.style.display = 'none';
            }
        }
    }

    // Reset filters
    function resetFilters() {
        console.log('Resetting filters');
        if (categoryFilter) categoryFilter.value = 'all';

        // Uncheck all implementation checkboxes
        implCheckboxes.forEach(checkbox => {
            checkbox.checked = false;
        });

        applyFilters();
    }

    // Attach event listeners
    if (categoryFilter) {
        categoryFilter.addEventListener('change', applyFilters);
        console.log('Category filter listener attached');
    }

    implCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', applyFilters);
    });
    console.log(`${implCheckboxes.length} checkbox listeners attached`);

    if (resetButton) {
        resetButton.addEventListener('click', resetFilters);
        console.log('Reset button listener attached');
    }

    // Initialize citations with metadata
    initializeCitations();
});
