/**
 * BiteID — Frontend Interactivity
 * Handles drag-and-drop upload, file preview, toggles, and animations.
 */

document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initAnimations();
});


// ============================================================
// Drag-and-Drop Upload
// ============================================================

function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const previewImg = document.getElementById('previewImg');
    const previewName = document.getElementById('previewName');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadForm = document.getElementById('uploadForm');

    if (!dropzone || !fileInput) return;

    // Click to open file dialog
    dropzone.addEventListener('click', (e) => {
        if (e.target === removeFileBtn || e.target.closest('#removeFileBtn')) return;
        fileInput.click();
    });

    // File selected via dialog
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            showPreview(fileInput.files[0]);
        }
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showPreview(files[0]);
        }
    });

    // Remove file
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearPreview();
        });
    }

    // Form submission — add loading state
    if (uploadForm) {
        uploadForm.addEventListener('submit', () => {
            if (analyzeBtn && !analyzeBtn.disabled) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<span class="spinner"></span> Analyzing...';
            }
        });
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) previewImg.src = e.target.result;
            if (previewName) previewName.textContent = file.name;
            if (dropzoneContent) dropzoneContent.style.display = 'none';
            if (dropzonePreview) dropzonePreview.style.display = 'flex';
            if (analyzeBtn) analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearPreview() {
        fileInput.value = '';
        if (dropzoneContent) dropzoneContent.style.display = 'flex';
        if (dropzonePreview) dropzonePreview.style.display = 'none';
        if (previewImg) previewImg.src = '';
        if (previewName) previewName.textContent = '';
        if (analyzeBtn) analyzeBtn.disabled = true;
    }
}


// ============================================================
// Toggle Sections (Debug Images, XAI)
// ============================================================

function toggleSection(sectionId, btn) {
    const section = document.getElementById(sectionId);
    if (!section) return;

    const isHidden = section.style.display === 'none';
    section.style.display = isHidden ? 'block' : 'none';

    if (btn) {
        const text = btn.textContent.trim();
        if (text.startsWith('Show')) {
            btn.textContent = text.replace('Show', 'Hide');
        } else if (text.startsWith('Hide')) {
            btn.textContent = text.replace('Hide', 'Show');
        }
    }

    // Animate entry
    if (isHidden) {
        section.style.opacity = '0';
        section.style.transform = 'translateY(-10px)';
        requestAnimationFrame(() => {
            section.style.transition = 'all 0.3s ease';
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
        });
    }
}


// ============================================================
// Scroll-triggered Animations
// ============================================================

function initAnimations() {
    // Animate elements on scroll into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    // Observe cards and sections
    document.querySelectorAll('.glass-card, .step-card, .match-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        observer.observe(el);
    });

    // Nav scroll effect
    const nav = document.getElementById('mainNav');
    if (nav) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 20) {
                nav.style.background = 'rgba(10, 14, 23, 0.9)';
                nav.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
            } else {
                nav.style.background = 'rgba(15, 23, 42, 0.65)';
                nav.style.boxShadow = 'none';
            }
        });
    }
}

// CSS class for animation completion
document.head.insertAdjacentHTML('beforeend', `
<style>
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
</style>
`);
