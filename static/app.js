// Dark mode toggle
const toggle = document.getElementById('theme-toggle');
function setTheme(theme) {
  document.body.classList.toggle('dark', theme === 'dark');
  localStorage.setItem('theme', theme);
  // change button text based on current theme
  toggle.textContent = theme === 'dark'
    ? 'Light'   // when in dark mode, offer “Light”
    : 'Dark';   // when in light mode, offer “Dark”
}

// Initialize theme
const saved = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
setTheme(saved);

toggle.addEventListener('click', () => {
  const next = document.body.classList.contains('dark') ? 'light' : 'dark';
  setTheme(next);
});

// AJAX form submission & prediction handling
const form = document.getElementById('upload-form');
const resultsDiv = document.getElementById('results');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const loading = document.getElementById('loading-overlay');
  loading.classList.remove('hidden');

  resultsDiv.innerHTML = '';
  resultsDiv.classList.remove('hidden');

  const formData = new FormData(form);

  // Start fetch but don't await yet
  const fetchPromise = fetch('/', { method: 'POST', body: formData })
    .then(resp => resp.json())
    .catch(err => {
      console.error(err);
      return { error: true };
    });

  // Create a 3-second delay promise
  const delay = new Promise(resolve => setTimeout(resolve, 2600));

  // Wait for BOTH fetch and delay to complete
  const [data] = await Promise.all([fetchPromise, delay]);

  // Hide loading overlay after 3 seconds AND fetch finish
  loading.classList.add('hidden');

  // Handle error or no results
  if (!data || data.error) {
    resultsDiv.innerHTML = '<p class="note highlight">Error processing image.</p>';
    return;
  }

  const results = data.results || [];
  if (results.length === 0) {
    resultsDiv.innerHTML = '<p class="note highlight">No faces detected – cannot predict.</p>';
    return;
  }

  // Render each result
  results.forEach((res) => {
    const card = document.createElement('div');
    card.className = 'result-card';

    const img = document.createElement('img');
    img.src = res.face;
    card.appendChild(img);

    const probs = document.createElement('div');
    probs.className = 'probs-container';
    Object.entries(res.probabilities).forEach(([emotion, prob]) => {
      const pEl = document.createElement('p');
      pEl.textContent = `${emotion}: ${Math.round(prob * 100)}%`;
      probs.appendChild(pEl);
    });
    card.appendChild(probs);

    // highlight highest
    const pEls = probs.querySelectorAll('p');
    let best = { el: null, value: -Infinity };
    pEls.forEach(el => {
      const p = parseFloat(el.textContent.split(':')[1]);
      if (p > best.value) best = { el, value: p };
    });
    if (best.el) best.el.classList.add('highlight');

    resultsDiv.appendChild(card);
  });
});


// On every page load, clear the form and hide any old results
window.addEventListener('DOMContentLoaded', () => {
  const form    = document.getElementById('upload-form');
  const results = document.getElementById('results');

  // Reset file‐input (and any other form fields)
  form.reset();

  // Hide previous results container
  results.classList.add('hidden');
});
