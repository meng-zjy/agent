/* 身份、文档和授权均由后端验证；页面隐藏仅用于减少误操作。 */
const select = (selector) => document.querySelector(selector);
let currentUser;
let accounts = [];
let viewVersion = 0;

async function requestKnowledge(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  if (response.status === 401) {
    window.location.assign('/');
    throw new Error('请先登录');
  }
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || '请求失败');
  return body;
}

async function perform(action) {
  select('#status').textContent = '正在处理…';
  try {
    await action();
    select('#status').textContent = '已完成';
  } catch (error) {
    select('#status').textContent = error.message;
  }
}

function baseUrl(baseId = select('#bases').value) {
  return '/api/knowledge-bases/' + encodeURIComponent(baseId);
}

async function loadBases(preferredId) {
  const bases = await requestKnowledge('/api/knowledge-bases');
  const chooser = select('#bases');
  chooser.replaceChildren();
  for (const base of bases) {
    const option = document.createElement('option');
    option.value = base.id;
    option.textContent = base.name + (base.public ? '（公开）' : '（受限）');
    chooser.append(option);
  }
  if (preferredId && bases.some(base => base.id === preferredId)) chooser.value = preferredId;
  await loadSelectedBase();
}

async function loadSelectedBase() {
  const version = ++viewVersion;
  const selectedId = select('#bases').value;
  const url = baseUrl(selectedId);
  select('#documents').replaceChildren();
  select('#preview').textContent = '';
  select('#grants').replaceChildren();
  select('#put-document').reset();
  if (!selectedId) {
    select('#documents').textContent = '暂无可访问的知识库，请联系管理员授权。';
    return;
  }
  const documents = await requestKnowledge(url + '/documents');
  if (version !== viewVersion) return;
  if (!documents.length) select('#documents').textContent = '暂无文档';
  for (const documentInfo of documents) {
    const item = document.createElement('li');
    const preview = document.createElement('button');
    preview.textContent = documentInfo.name;
    preview.onclick = () => perform(async () => {
      const documentBody = await requestKnowledge(url + '/documents/' + encodeURIComponent(documentInfo.id));
      if (version !== viewVersion) return;
      select('#preview').textContent = documentBody.content;
      if (currentUser.is_admin) {
        select('#document-name').value = documentBody.name;
        select('#document-content').value = documentBody.content;
      }
    });
    item.append(preview);
    if (currentUser.is_admin) {
      const remove = document.createElement('button');
      remove.textContent = '删除';
      remove.onclick = () => {
        if (!confirm('确认删除文档“' + documentInfo.name + '”？')) return;
        perform(async () => {
          await requestKnowledge(url + '/documents/' + encodeURIComponent(documentInfo.id), { method: 'DELETE' });
          await loadSelectedBase();
        });
      };
      item.append(remove);
    }
    select('#documents').append(item);
  }
  if (!currentUser.is_admin) return;
  const grants = await requestKnowledge(url + '/grants');
  if (version !== viewVersion) return;
  for (const account of accounts) {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('span');
    label.textContent = account.username;
    if (account.is_admin) {
      label.textContent += '（管理员，自动拥有访问权限）';
      row.append(label);
      select('#grants').append(row);
      continue;
    }
    const granted = grants.includes(account.id);
    const button = document.createElement('button');
    button.textContent = granted ? '撤销授权' : '授予访问';
    button.onclick = () => perform(async () => {
      button.disabled = true;
      try {
        await requestKnowledge(url + '/grants/' + account.id, { method: granted ? 'DELETE' : 'PUT' });
        await loadSelectedBase();
      } finally {
        button.disabled = false;
      }
    });
    row.append(label, button);
    select('#grants').append(row);
  }
}

select('#bases').onchange = () => perform(loadSelectedBase);
select('#create-base').onsubmit = (event) => {
  event.preventDefault();
  perform(async () => {
    const base = await requestKnowledge('/api/knowledge-bases', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: select('#base-name').value, public: select('#base-public').checked }),
    });
    select('#create-base').reset();
    await loadBases(base.id);
  });
};
select('#put-document').onsubmit = (event) => {
  event.preventDefault();
  perform(async () => {
    if (!select('#bases').value) throw new Error('请先创建或选择知识库');
    await requestKnowledge(baseUrl() + '/documents', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: select('#document-name').value, content: select('#document-content').value }),
    });
    select('#put-document').reset();
    await loadSelectedBase();
  });
};

perform(async () => {
  const account = await requestKnowledge('/api/auth/me');
  if (!account.enabled || !account.user) {
    window.location.assign('/');
    return;
  }
  currentUser = account.user;
  select('#identity').textContent = currentUser.username + (currentUser.is_admin ? ' · 管理员' : '');
  select('#admin').hidden = !currentUser.is_admin;
  if (currentUser.is_admin) accounts = await requestKnowledge('/api/admin/users');
  await loadBases();
});
