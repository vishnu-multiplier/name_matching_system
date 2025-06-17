// Tabs functionality
const tabs = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const selectedTab = tab.getAttribute('data-tab');

    // Set aria-selected and styling
    tabs.forEach(t => {
      t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
      t.classList.toggle('border-blue-600', t === tab);
      t.classList.toggle('text-blue-600', t === tab);
      t.classList.toggle('text-gray-600', t !== tab);
    });

    // Show/hide tab content
    tabContents.forEach(content => {
      content.classList.toggle('hidden', content.id !== selectedTab);
    });
  });
});

// Drag & Drop + File Input handler helper
function setupDragDrop(areaId, inputId, fileNameId) {
  const dropArea = document.getElementById(areaId);
  const fileInput = document.getElementById(inputId);
  const fileNameDisplay = document.getElementById(fileNameId);

  // Highlight on dragover
  dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.classList.add('dragover');
  });

  dropArea.addEventListener('dragleave', e => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
  });

  dropArea.addEventListener('drop', e => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      updateFileName();
    }
  });

  // **Add mouse click handler**
  dropArea.addEventListener('click', e => {
    fileInput.click();
  });

  // Keyboard accessible
  dropArea.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  // When user selects file via dialog
  fileInput.addEventListener('change', updateFileName);

  function updateFileName() {
    if (fileInput.files.length > 0) {
      fileNameDisplay.textContent = `Selected file: ${fileInput.files[0].name}`;
    } else {
      fileNameDisplay.textContent = '';
    }
  }
}


setupDragDrop('trainDropArea', 'trainFile', 'trainFileName');
setupDragDrop('testDropArea', 'testFile', 'testFileName');

// Utility: display error
function displayError(element, message) {
  element.textContent = `Error: ${message}`;
}


// Handle Train form submission
document.getElementById('trainForm').addEventListener('submit', async e => {
  e.preventDefault();

  const trainFileInput = document.getElementById('trainFile');
  if (trainFileInput.files.length === 0) {
    showToast('Please select a training CSV file.', 'information');
    return;
  }

  const formData = new FormData();
  formData.append('train_file', trainFileInput.files[0]);

  const trainingOutput = document.getElementById('trainingOutput');
  const trainingProgress = document.getElementById('trainingProgress');
  trainingOutput.classList.remove('hidden');
  trainingProgress.textContent = 'Starting training...';

  try {
    const response = await fetch('/train', {
      method: 'POST',
      body: formData,
    });

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      displayError(trainingProgress, `Unexpected response format:\n${text}`);
      return;
    }

    const data = await response.json();
    console.log('Training response:', data);

    if (response.ok && data?.status === 'success') {
      trainingProgress.textContent = data.message || 'Training completed successfully!';
    } else {
      displayError(trainingProgress, data?.error || 'Unexpected response format.');
    }
  } catch (error) {
    displayError(trainingProgress, error.message);
  }
});

// Handle Test form submission
document.getElementById('testForm').addEventListener('submit', async e => {
  e.preventDefault();

  const testFileInput = document.getElementById('testFile');
  if (testFileInput.files.length === 0) {
    showToast('Please select a test CSV file.', 'information');
    return;
  }

  const formData = new FormData();
  formData.append('test_file', testFileInput.files[0]);

  const testingOutput = document.getElementById('testingOutput');
  const testingProgress = document.getElementById('testingProgress');
  const downloadSection = document.getElementById('downloadSection');
  const downloadLinks = document.getElementById('downloadLinks');

  testingOutput.classList.remove('hidden');
  downloadSection.classList.add('hidden');
  testingProgress.textContent = 'Starting testing...';

  try {
    const response = await fetch('/test', {
      method: 'POST',
      body: formData,
    });

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      displayError(testingProgress, `Unexpected response format:\n${text}`);
      return;
    }

    const data = await response.json();
    console.log('Testing response:', data);

    if (response.ok && data?.status === 'success') {
      let finalMessage = data.message || 'Testing completed successfully!';

      // ⛔️ Handle merging error gracefully
      if (data?.merging?.status === 'error') {
        finalMessage += `\n Merging failed: ${data.merging.error_message}`;
        showToast(`${data.merging.error_message}`, 'warning');
      }

      testingProgress.textContent = finalMessage;

      // ✅ Display download links if available
      if (data.downloads && Object.keys(data.downloads).length > 0) {
        downloadSection.classList.remove('hidden');
        downloadLinks.innerHTML = '';
        for (const [name, url] of Object.entries(data.downloads)) {
          const link = document.createElement('a');
          link.href = url;
          link.download = '';
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.className = 'inline-block bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition';
          link.textContent = name;
          downloadLinks.appendChild(link);
        }
      } else {
        downloadSection.classList.add('hidden');
        downloadLinks.innerHTML = '';
      }
    } else {
      displayError(testingProgress, data?.error || 'Unexpected response format.');
    }
  } catch (error) {
    displayError(testingProgress, error.message);
  }
});


document.getElementById('clearCacheButton').addEventListener('click', function() {
  
    fetch('/clear_cache', { method: 'GET' })
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          showToast(data.message, 'success');
        } else {
          showToast(data.message, 'danger');
        }
      })
      .catch(error => {
        console.error('Error clearing cache:', error);
        showToast('An error occurred while clearing the cache.','danger');
      });
  
})


    // Track login state for train tab
    let isTrainLoggedIn = false;

    document.addEventListener('DOMContentLoaded', function() {
      const trainTab = document.getElementById('train-tab');
      const tabContents = document.querySelectorAll('.tab-content');
      const tabsNav = document.getElementById('tabsNav');
      const trainLockDialog = document.getElementById('trainLockDialog');
      const trainTabContent = document.getElementById('trainTabContent');

      // Show login dialog only for train tab if not logged in
      tabsNav.addEventListener('click', function(e) {
        if (e.target.classList.contains('tab-button')) {
          const tab = e.target.getAttribute('data-tab');
          // Switch tabs
          document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.toggle('text-blue-600', btn.getAttribute('data-tab') === tab);
            btn.classList.toggle('border-blue-600', btn.getAttribute('data-tab') === tab);
            btn.classList.toggle('text-gray-600', btn.getAttribute('data-tab') !== tab);
            btn.classList.toggle('border-transparent', btn.getAttribute('data-tab') !== tab);
          });
          tabContents.forEach(section => {
            section.classList.toggle('hidden', section.id !== tab);
          });

          // If train tab, check login
          if (tab === 'train') {
            if (!isTrainLoggedIn) {
              trainLockDialog.style.display = 'flex';
              trainTabContent.style.display = 'none';
            } else {
              trainLockDialog.style.display = 'none';
              trainTabContent.style.display = '';
            }
          }
        }
      });

      // On page load, if train tab is default, show lock if not logged in
      if (!isTrainLoggedIn && document.getElementById('train').classList.contains('tab-content') && !document.getElementById('train').classList.contains('hidden')) {
        trainLockDialog.style.display = 'flex';
        trainTabContent.style.display = 'none';
      }

      // Train login form logic
      document.getElementById('trainLoginForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const username = document.getElementById('train-username').value.trim();
        const password = document.getElementById('train-password').value.trim();
        if (username === 'admin' && password === 'admin') {
          isTrainLoggedIn = true;
          trainLockDialog.style.display = 'none';
          trainTabContent.style.display = '';
        } else {
          document.getElementById('trainLoginError').classList.remove('hidden');
        }
      });
    });



  // Custom Toast Notification

let icon = {
    success:
    '<span class="material-symbols-outlined">task_alt</span>',
    danger:
    '<span class="material-symbols-outlined">error</span>',
    warning:
    '<span class="material-symbols-outlined">warning</span>',
    info:
    '<span class="material-symbols-outlined">info</span>',
};

const showToast = (
    message,
    toastType,
    duration = 10000) => {
    if (
        !Object.keys(icon).includes(toastType))
        toastType = "info";

    let box = document.createElement("div");
    box.classList.add(
        "toast", `toast-${toastType}`);
    box.innerHTML = ` <div class="toast-content-wrapper">
                      <div class="toast-icon">
                      ${icon[toastType]}
                      </div>
                      <div class="toast-message">${message}</div>
                      <div class="toast-progress"></div>
                      </div>`;
    duration = duration;
    box.querySelector(".toast-progress").style.animationDuration =
            `${duration / 10000}s`;

    let toastAlready = 
        document.body.querySelector(".toast");
    if (toastAlready) {
        toastAlready.remove();
    }

    document.body.appendChild(box)};