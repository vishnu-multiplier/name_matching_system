// Tabs functionality
const tabs = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const selectedTab = tab.getAttribute('data-tab');

    tabs.forEach(t => {
      t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
      t.classList.toggle('border-blue-600', t === tab);
      t.classList.toggle('text-blue-600', t === tab);
      t.classList.toggle('text-gray-600', t !== tab);
    });

    tabContents.forEach(content => {
      content.classList.toggle('hidden', content.id !== selectedTab);
    });
  });
});

// Drag & Drop + File Input helper
function setupDragDrop(areaId, inputId, fileNameId) {
  const dropArea = document.getElementById(areaId);
  const fileInput = document.getElementById(inputId);
  const fileNameDisplay = document.getElementById(fileNameId);

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

  dropArea.addEventListener('click', () => fileInput.click());

  dropArea.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

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

function displayError(element, message) {
  element.textContent = `Error: ${message}`;
}

// Train form submission
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
    const response = await fetch('/train', { method: 'POST', body: formData });
    const data = await response.json();

    if (response.ok && data?.status === 'success') {
      trainingProgress.textContent = data.message || 'Training completed successfully!';
    } else {
      displayError(trainingProgress, data?.error || 'Unexpected response.');
    }
  } catch (error) {
    displayError(trainingProgress, error.message);
  }
});

// Test form submission with Excel handling
document.getElementById('testForm').addEventListener('submit', async e => {
  e.preventDefault();
  const testFileInput = document.getElementById('testFile');
  if (testFileInput.files.length === 0) {
    showToast('Please select a test file.', 'information');
    return;
  }

  const formData = new FormData();
  formData.append('test_file', testFileInput.files[0]);

  const testingOutput = document.getElementById('testingOutput');
  const testingProgress = document.getElementById('testingProgress');
  const downloadSection = document.getElementById('downloadSection');
  const downloadLinks = document.getElementById('downloadLinks');
  const sheetSelectionSection = document.getElementById('sheetSelectionSection');
  const sheetDropdown = document.getElementById('sheetDropdown');

  testingOutput.classList.remove('hidden');
  downloadSection.classList.add('hidden');
  sheetSelectionSection.classList.add('hidden');
  testingProgress.textContent = 'Starting testing...';

  try {
    const response = await fetch('/test', { method: 'POST', body: formData });
    const data = await response.json();

    if (response.ok && data?.status === 'success') {
      handleTestSuccess(data);
    } else if (response.ok && data?.status === 'excel') {
      displaySheetSelection(data);
    } else {
      displayError(testingProgress, data?.error || 'Unexpected response.');
    }
  } catch (error) {
    displayError(testingProgress, error.message);
  }
});

function handleTestSuccess(data) {
  const testingProgress = document.getElementById('testingProgress');
  const downloadSection = document.getElementById('downloadSection');
  const downloadLinks = document.getElementById('downloadLinks');

  let finalMessage = data.message || 'Testing completed successfully!';
  if (data?.merging?.status === 'error') {
    finalMessage += `\n Merging failed: ${data.merging.message}`;
    showToast(data.merging.message, 'warning');
  }
  testingProgress.textContent = finalMessage;

  if (data.downloads && Object.keys(data.downloads).length > 0) {
    downloadSection.classList.remove('hidden');
    downloadLinks.innerHTML = '';
    for (const [name, url] of Object.entries(data.downloads)) {
      const link = document.createElement('a');
      link.href = url;
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
}

function displaySheetSelection(data) {
  const sheetSelectionSection = document.getElementById('sheetSelectionSection');
  const sheetDropdown = document.getElementById('sheetDropdown');
  const testingProgress = document.getElementById('testingProgress');

  sheetDropdown.innerHTML = '';
  data.sheets.forEach(sheet => {
    const option = document.createElement('option');
    option.value = sheet;
    option.textContent = sheet;
    sheetDropdown.appendChild(option);
  });

  sheetSelectionSection.classList.remove('hidden');
  testingProgress.textContent = 'Please select a sheet to process.';

  document.getElementById('processSheetButton').onclick = () => {
    const selectedSheet = sheetDropdown.value;
    if (!selectedSheet) {
      showToast('Please select a sheet.', 'information');
      return;
    }
    processSelectedSheet(data.filename, selectedSheet);
  };
}

async function processSelectedSheet(filename, sheetName) {
  const testingProgress = document.getElementById('testingProgress');
  const sheetSelectionSection = document.getElementById('sheetSelectionSection');
  const downloadSection = document.getElementById('downloadSection');
  const downloadLinks = document.getElementById('downloadLinks');

  sheetSelectionSection.classList.add('hidden');
  testingProgress.textContent = 'Processing selected sheet...';
  downloadSection.classList.add('hidden');

  const formData = new FormData();
  formData.append('filename', filename);
  formData.append('sheet_name', sheetName);

  try {
    const response = await fetch('/process_excel', { method: 'POST', body: formData });
    const data = await response.json();

    if (response.ok && data?.status === 'success') {
      handleTestSuccess(data);
    } else {
      displayError(testingProgress, data?.error || 'Unexpected response.');
    }
  } catch (error) {
    displayError(testingProgress, error.message);
  }
}

// Clear cache
document.getElementById('clearCacheButton').addEventListener('click', () => {
  fetch('/clear_cache', { method: 'GET' })
    .then(res => res.json())
    .then(data => {
      showToast(data.message, data.status === 'success' ? 'success' : 'danger');
    })
    .catch(() => {
      showToast('An error occurred while clearing the cache.', 'danger');
    });
});

// Train tab login control
let isTrainLoggedIn = false;
document.addEventListener('DOMContentLoaded', () => {
  const tabsNav = document.getElementById('tabsNav');
  const tabContents = document.querySelectorAll('.tab-content');
  const trainLockDialog = document.getElementById('trainLockDialog');
  const trainTabContent = document.getElementById('trainTabContent');

  // Show login dialog if train tab is active and not logged in
  const activeTab = document.querySelector('.tab-button[aria-selected="true"]');
  if (activeTab && activeTab.getAttribute('data-tab') === 'train' && !isTrainLoggedIn) {
    trainLockDialog.style.display = 'flex';
    trainTabContent.style.display = 'none';
  }

  // Always hide trainTabContent if not logged in
  if (!isTrainLoggedIn) {
    trainLockDialog.style.display = 'flex';
    trainTabContent.style.display = 'none';
  }

  tabsNav.addEventListener('click', e => {
    if (e.target.classList.contains('tab-button')) {
      const tab = e.target.getAttribute('data-tab');

      document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.toggle('text-blue-600', btn.getAttribute('data-tab') === tab);
        btn.classList.toggle('border-blue-600', btn.getAttribute('data-tab') === tab);
      });

      tabContents.forEach(section => {
        section.classList.toggle('hidden', section.id !== tab);
      });

      if (tab === 'train') {
        if (!isTrainLoggedIn) {
          trainLockDialog.style.display = 'flex';
          trainTabContent.style.display = 'none';
        } else {
          trainLockDialog.style.display = 'none';
          trainTabContent.style.display = '';
        }
      } else {
        // Hide login dialog and train content when not on train tab
        trainLockDialog.style.display = 'none';
        trainTabContent.style.display = 'none';
      }
    }
  });

  document.getElementById('trainLoginForm').addEventListener('submit', e => {
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

// Toast Notification
const icon = {
  success: '<span class="material-symbols-outlined">task_alt</span>',
  danger: '<span class="material-symbols-outlined">error</span>',
  warning: '<span class="material-symbols-outlined">warning</span>',
  information: '<span class="material-symbols-outlined">info</span>',
};

function showToast(message, toastType = 'information', duration = 10000) {
  if (!icon[toastType]) toastType = 'information';

  const box = document.createElement('div');
  box.classList.add('toast', `toast-${toastType}`);
  box.innerHTML = `
    <div class="toast-content-wrapper">
      <div class="toast-icon">${icon[toastType]}</div>
      <div class="toast-message">${message}</div>
      <div class="toast-progress"></div>
    </div>
  `;
  box.querySelector('.toast-progress').style.animationDuration = `${duration / 1000}s`;

  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  document.body.appendChild(box);
}
