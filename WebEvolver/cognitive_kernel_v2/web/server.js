const express = require('express');
// const { chromium } = require('playwright');
const { chromium } = require('playwright-extra')
const StealthPlugin = require('puppeteer-extra-plugin-stealth')
const { v4: uuidv4 } = require('uuid'); 
const yaml = require('js-yaml');
const fs = require('fs').promises; 
const path = require('path');

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
const app = express();
const port = 3000;

app.use(express.json());

let browserPool = {};
const maxBrowsers = parseInt(process.env.MAX_BROWSERS) || 16;
let waitingQueue = [];
let pageIdToChunkedTree = {};
let pageIdToChunkedContent = {};

const initializeBrowserPool = (size) => {
  for (let i = 0; i < size; i++) {
    browserPool[String(i)] = {
      browserId: null,
      status: 'empty', 
      browser: null, 
      pages: {}, 
      lastActivity: Date.now() 
    };
  }
};

const v8 = require('v8');

// 将字节转换为更易读的格式
const formatBytes = (bytes) => {
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  if (bytes === 0) return '0 Byte';
  const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)), 10);
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
};

const logMemoryUsage = () => {
  const memoryUsage = process.memoryUsage();
  const heapStats = v8.getHeapStatistics();

  console.log('Memory Usage:');
  console.log(`  RSS: ${formatBytes(memoryUsage.rss)} (常驻内存)`);
  console.log(`  Heap Total: ${formatBytes(memoryUsage.heapTotal)} (堆的总大小)`);
  console.log(`  Heap Used: ${formatBytes(memoryUsage.heapUsed)} (使用的堆内存)`);
  console.log(`  External: ${formatBytes(memoryUsage.external)} (外部内存)`);
  
  console.log('Heap Statistics:');
  console.log(`  Total Heap Size: ${formatBytes(heapStats.total_heap_size)} (总堆大小)`);
  console.log(`  Total Heap Size Executable: ${formatBytes(heapStats.total_heap_size_executable)} (可执行堆大小)`);
  console.log(`  Total Physical Size: ${formatBytes(heapStats.total_physical_size)} (总物理大小)`);
  console.log(`  Total Available Size: ${formatBytes(heapStats.total_available_size)} (可用总大小)`);
  console.log(`  Used Heap Size: ${formatBytes(heapStats.used_heap_size)} (已用堆大小)`);
  console.log(`  Heap Size Limit: ${formatBytes(heapStats.heap_size_limit)} (堆大小限制)`);
};

const processNextInQueue = async () => {
  const availableBrowserslot = Object.keys(browserPool).find(
    id => browserPool[id].status === 'empty'
  );

  if (waitingQueue.length > 0 && availableBrowserslot) {
    const nextRequest = waitingQueue.shift();
    try {
      const browserEntry = browserPool[availableBrowserslot];
      let browserId = uuidv4()
      browserEntry.browserId = browserId
      // if (!browserEntry.browser) {
      //   const new_browser = await chromium.launch({ headless: true });
      //   browserEntry.browser = await new_browser.newContext({viewport: {width: 1024, height: 768}});
      // }
      browserEntry.status = 'not'; 
      nextRequest.res.send({ availableBrowserslot: availableBrowserslot });
    } catch (error) {
      nextRequest.res.status(500).send({ error: 'Failed to allocate browser.' });
    }
  } else if (waitingQueue.length > 0) {

  }
};


const releaseBrowser = async (browserslot) => {
  const browserEntry = browserPool[browserslot];
  if (browserEntry && browserEntry.browser) {
    await browserEntry.browser.close();
    browserEntry.browserId = null;
    browserEntry.status = 'empty';
    browserEntry.browser = null;
    browserEntry.pages = {};
    browserEntry.lastActivity = Date.now(); 

    processNextInQueue();
  }
};

setInterval(async () => {
  const now = Date.now();
  for (const [browserslot, browserEntry] of Object.entries(browserPool)) {
    if (browserEntry.status === 'not' && now - browserEntry.lastActivity > 600000) {
      await releaseBrowser(browserslot);
    }
  }
}, 30000); 

function findPageByPageId(browserId, pageId) {
  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (browserEntry && browserEntry.pages[pageId]) {
    return browserEntry.pages[pageId];
  }
  return null; 
}

function findPagePrefixesWithCurrentMark(browserId, currentPageId) {
  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  let pagePrefixes = [];

  if (browserEntry) {
    console.log(`current page id:${currentPageId}`, typeof currentPageId)
    for (const pageId in browserEntry.pages) {
      
      const page = browserEntry.pages[pageId];
      const pageTitle = page.pageTitle; 
      console.log(`iter page id:${pageId}`, typeof pageId)
      const isCurrentPage = pageId === currentPageId;
      const pagePrefix = `Tab ${pageId}${isCurrentPage ? ' (current)' : ''}: ${pageTitle}`;

      pagePrefixes.push(pagePrefix);
    }
  }

  return pagePrefixes.length > 0 ? pagePrefixes.join('\n') : null;
}

const { Mutex } = require('async-mutex');
const mutex = new Mutex();

app.post('/getBrowser', async (req, res) => {
  const { storageState, geoLocation } = req.body;

  const tryAllocateBrowser = () => {
    const availableBrowserslot = Object.keys(browserPool).find(
      id => browserPool[id].status === 'empty'
    );
    let browserId = null;
    if (availableBrowserslot) {
      browserId = uuidv4();
      browserPool[availableBrowserslot].browserId = browserId;
    }
    return { availableBrowserslot, browserId };
  };

  const waitForAvailableBrowser = () => {
    return new Promise(resolve => {
      waitingQueue.push(request => resolve(request));
    });
  };

  // Acquire the mutex lock
  const release = await mutex.acquire();

  try {
    let { availableBrowserslot, browserId } = tryAllocateBrowser();
    if (!availableBrowserslot) {
      await waitForAvailableBrowser().then((id) => {
        availableBrowserslot = id;
      });
    }
    console.log(storageState);
    let browserEntry = browserPool[availableBrowserslot];
    if (!browserEntry.browser) {
      chromium.use(StealthPlugin());
      const new_browser = await chromium.launch({ headless: true });
      if (storageState) {
        browserEntry.browser = await new_browser.newContext({
          viewport: { width: 1024, height: 768 },
          storageState: storageState,
          geolocation: geoLocation
        });
      } else {
        browserEntry.browser = await new_browser.newContext({
          viewport: { width: 1024, height: 768 }
        });
      }
    }
    browserEntry.status = 'not';
    browserEntry.lastActivity = Date.now();
    console.log(`browserId: ${browserId}`);
    res.send({ browserId: browserId });
  } finally {
    // Release the mutex lock
    release();
  }
});

app.post('/closeBrowser', async (req, res) => {
  const { browserId } = req.body;

  if (!browserId) {
    return res.status(400).send({ error: 'Missing required field: browserId.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot] 
  if (!browserEntry || !browserEntry.browser) {
    return res.status(404).send({ error: 'Browser not found.' });
  }

  try {
    await browserEntry.browser.close();

    browserEntry.browserId = null;
    browserEntry.pages = {};
    browserEntry.browser = null;
    browserEntry.status = 'empty';
    browserEntry.lastActivity = null;

    if (waitingQueue.length > 0) {
      const nextRequest = waitingQueue.shift();
      const nextAvailableBrowserId = Object.keys(browserPool).find(
        id => browserPool[id].status === 'empty'
      );
      if (nextRequest && nextAvailableBrowserId) {
        browserPool[nextAvailableBrowserId].status = 'not';
        nextRequest(nextAvailableBrowserId);
      }
    }

    res.send({ message: 'Browser closed successfully.' });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to close browser.' });
  }
});

const close_all_browsers = async () => {
  // Acquire the mutex lock
  const release = await mutex.acquire();

  try {
    for (const slot in browserPool) {
      const browserEntry = browserPool[slot];
      if (browserEntry.browser) {
        try {
          console.error(`closing browserid`, browserEntry.browserId);
          await browserEntry.browser.close();
          browserEntry.browserId = null;
          browserEntry.pages = {};
          browserEntry.browser = null;
          browserEntry.status = 'empty';
          browserEntry.lastActivity = null;
        } catch (error) {
          console.error(`Failed to close browser in slot ${slot}:`, error);
        }
      }
    }

    // Clear the waiting queue
    while (waitingQueue.length > 0) {
      const nextRequest = waitingQueue.shift();
      if (nextRequest) {
        nextRequest(null); // Resolve the promise with null to indicate no available browser
      }
    }

    console.log('All browsers closed successfully.');
  } finally {
    // Release the mutex lock
    release();
  }
};

// Example usage
app.post('/closeAllBrowsers', async (req, res) => {
  try {
    await close_all_browsers();
    res.send({ message: 'All browsers closed successfully.' });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to close all browsers.' });
  }
});

app.post('/openPage', async (req, res) => {
  const { browserId, url } = req.body;

  if (!browserId || !url) {
    return res.status(400).send({ error: 'Missing browserId or url.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  // const browserEntry = browserPool[browserId];
  if (!browserEntry || !browserEntry.browser) {
    return res.status(404).send({ error: 'Browser not found.' });
  }
  console.log(await browserEntry.browser.storageState());
  const setCustomUserAgent = async (page) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'userAgent', {
        get: () => 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      });
    });
  };

  const release = await mutex.acquire();
  try {
    const page = await browserEntry.browser.newPage();
    await setCustomUserAgent(page);
    await page.goto(url);
    const pageIdint = Object.keys(browserEntry.pages).length;
    console.log(`current page id:${pageIdint}`)
    const pageTitle = await page.title();
    const pageId = String(pageIdint);
    browserEntry.pages[pageId] = {'pageId': pageId, 'pageTitle': pageTitle, 'page': page, 'downloadedFiles': [], 'downloadSources': []}; 
    browserEntry.lastActivity = Date.now(); 

    // Define your download path
    const downloadPath = '/app/DownloadedFiles';
    path.resolve(downloadPath);
    console.log(`Download path: ${downloadPath}`);

    // Ensure the download directory exists
    try {
      await fs.access(downloadPath);
    } catch (error) {
      if (error.code === 'ENOENT') {
        await fs.mkdir(downloadPath, { recursive: true });
      } else {
        console.error(`Failed to access download directory: ${error}`);
        return;
      }
    }

    // Listen for the download event
    page.on('download', async (download) => {
      try {
        console.log('Download object properties:', download.url(), download.suggestedFilename(), download.failure());
        const tmp_downloadPath = await download.path();
        console.log(`Download path: ${tmp_downloadPath}`);
        // Get the original filename
        const filename = download.suggestedFilename();
        console.log(`Suggested filename: ${filename}`);
        // Create the full path to save the file
        const filePath = path.join(downloadPath, filename);
        console.log(`Saving to path: ${filePath}`);
        // Save the file to the specified path
        await download.saveAs(filePath);
        console.log(`Download completed: ${filePath}`);
        browserEntry.pages[pageId].downloadedFiles.push(filePath);
      } catch (error) {
        console.error(`Failed to save download: ${error}`);
      }
    });

      // Function to set user agent on a page
    
    // const tmp_page = await browserEntry.browser.newPage();
    // await setCustomUserAgent(tmp_page);
    // await tmp_page.goto('https://www.google.com');
    // const tmp_user_agent = await tmp_page.evaluate(() => navigator.userAgent);
    // console.log('USER AGENT after creation: ', tmp_user_agent);
    
    const userAgent = await page.evaluate(() => navigator.userAgent);
    console.log('USER AGENT: ', userAgent);

    res.send({ browserId, pageId });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to open new page.' });
  }
  finally {
    // Release the mutex lock
    release();
  }
});

function parseAccessibilityTree(nodes) {
  const IGNORED_ACTREE_PROPERTIES = [
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "hiddenRoot",
    "hidden",
    "controls",
    "labelledby",
    "describedby",
    "url"
  ];
  const IGNORED_ACTREE_ROLES = [
    "gridcell",
  ];
  
  let nodeIdToIdx = {};
  nodes.forEach((node, idx) => {
    if (!(node.nodeId in nodeIdToIdx)) {
      nodeIdToIdx[node.nodeId] = idx;
    }
  });
  let treeIdxtoElement = {};
  function dfs(idx, depth, parent_name) {
    let treeStr = "";
    let node = nodes[idx];
    let indent = "\t".repeat(depth);
    let validNode = true;
    try {

      let role = node.role.value;
      let name = node.name.value;
      let nodeStr = `${role} '${name}'`;
      if (!name.trim() || IGNORED_ACTREE_ROLES.includes(role) || (parent_name.trim().includes(name.trim()) && ["StaticText", "heading", "image", "generic"].includes(role))){
        validNode = false;
      } else{
        let properties = [];
        (node.properties || []).forEach(property => {
          if (!IGNORED_ACTREE_PROPERTIES.includes(property.name)) {
            properties.push(`${property.name}: ${property.value.value}`);
          }
        });

        if (properties.length) {
          nodeStr += " " + properties.join(" ");
        }
      }

      if (validNode) {
        treeIdxtoElement[Object.keys(treeIdxtoElement).length + 1] = node;
        treeStr += `${indent}[${Object.keys(treeIdxtoElement).length}] ${nodeStr}`;
      }
    } catch (e) {
      validNode = false;
    }
    for (let childNodeId of node.childIds) {
      if (Object.keys(treeIdxtoElement).length >= 300) {
        break; 
      }
      
      if (!(childNodeId in nodeIdToIdx)) {
        continue; 
      }
    
      let childDepth = validNode ? depth + 1 : depth;
      let curr_name = validNode ? node.name.value : parent_name;
      let childStr = dfs(nodeIdToIdx[childNodeId], childDepth, curr_name);
      if (childStr.trim()) {
        if (treeStr.trim()) {
          treeStr += "\n";
        }
        treeStr += childStr;
      }
    }
    return treeStr;
  }

  let treeStr = dfs(0, 0, 'root');
  return {treeStr, treeIdxtoElement};
}

async function getBoundingClientRect(client, backendNodeId) {
  try {
      // Resolve the node to get the RemoteObject
      const remoteObject = await client.send("DOM.resolveNode", {backendNodeId: parseInt(backendNodeId)});
      const remoteObjectId = remoteObject.object.objectId;

      // Call a function on the resolved node to get its bounding client rect
      const response = await client.send("Runtime.callFunctionOn", {
          objectId: remoteObjectId,
          functionDeclaration: `
              function() {
                  if (this.nodeType === 3) { // Node.TEXT_NODE
                      var range = document.createRange();
                      range.selectNode(this);
                      var rect = range.getBoundingClientRect().toJSON();
                      range.detach();
                      return rect;
                  } else {
                      return this.getBoundingClientRect().toJSON();
                  }
              }
          `,
          returnByValue: true
      });
      return response;
  } catch (e) {
      return {result: {subtype: "error"}};
  }
}

async function getLocatorFromNodeId(page, node) {
  try {
    //const locator = page.locator(`[backendDOMNodeId="${node.backendDOMNodeId}"]`);
    const locator = await page.evaluateHandle((nodeId) => {
      return document.querySelector(`[backendNodeId="${nodeId}"]`);
    }, node.backendDOMNodeId);
    return locator;
  } catch (e) {
    return null;
  }
}

/**
 * @param {import("puppeteer").Frame} frame
 * @param {number} backendNodeId
 */
async function resolveNodeFromBackendNodeId(frame, backendNodeId) {
  const ctx = await Promise.resolve(frame.executionContext())
  return /** @type {import('puppeteer').ElementHandle} */ (
    /** @type {any} */ (ctx)._adoptBackendNodeId(backendNodeId)
  )
}

async function fetchPageAccessibilityTree(accessibilityTree) {
  let seenIds = new Set();
  let filteredAccessibilityTree = [];
  let backendDOMids = [];
  for (let i = 0; i < accessibilityTree.length; i++) {
      if (filteredAccessibilityTree.length >= 20000) {
          break;
      }
      let node = accessibilityTree[i];
      if (!seenIds.has(node.nodeId) && 'backendDOMNodeId' in node) {
          filteredAccessibilityTree.push(node);
          seenIds.add(node.nodeId);
          backendDOMids.push(node.backendDOMNodeId);
      }
  }
  accessibilityTree = filteredAccessibilityTree;
  return [accessibilityTree, backendDOMids];
}

async function fetchAllBoundingClientRects(client, backendNodeIds) {
  const fetchRectPromises = backendNodeIds.map(async (backendNodeId) => {
      return getBoundingClientRect(client, backendNodeId);
  });

  try {
      const results = await Promise.all(fetchRectPromises);
      return results; 
  } catch (error) {
      console.error("An error occurred:", error);
  }
}

function removeNodeInGraph(node, nodeidToCursor, accessibilityTree) {
  const nodeid = node.nodeId;
  const nodeCursor = nodeidToCursor[nodeid];
  const parentNodeid = node.parentId;
  const childrenNodeids = node.childIds;
  const parentCursor = nodeidToCursor[parentNodeid];
  // Update the children of the parent node
  if (accessibilityTree[parentCursor] !== undefined) {
    // Remove the nodeid from parent's childIds
    const index = accessibilityTree[parentCursor].childIds.indexOf(nodeid);
    //console.log('index:', index);
    accessibilityTree[parentCursor].childIds.splice(index, 1);
    // Insert childrenNodeids in the same location
    childrenNodeids.forEach((childNodeid, idx) => {
      if (childNodeid in nodeidToCursor) {
        accessibilityTree[parentCursor].childIds.splice(index + idx, 0, childNodeid);
      }
    });
    // Update children node's parent
    childrenNodeids.forEach(childNodeid => {
      if (childNodeid in nodeidToCursor) {
        const childCursor = nodeidToCursor[childNodeid];
        accessibilityTree[childCursor].parentId = parentNodeid;
      }
    });
  }
  accessibilityTree[nodeCursor].parentId = "[REMOVED]";
}

function processAccessibilityTree(accessibilityTree) {
  const nodeidToCursor = {};
  accessibilityTree.forEach((node, index) => {
    nodeidToCursor[node.nodeId] = index;
  });
  let count = 0;
  accessibilityTree.forEach(node => {
    if (node.union_bound === undefined) {
      removeNodeInGraph(node, nodeidToCursor, accessibilityTree);
      return;
    }
    const x = node.union_bound.x;
    const y = node.union_bound.y;
    const width = node.union_bound.width;
    const height = node.union_bound.height;
    
    // Invisible node
    if (width === 0 || height === 0) {
      removeNodeInGraph(node, nodeidToCursor, accessibilityTree);
      return;
    }

    const inViewportRatio = getInViewportRatio(
      parseFloat(x),
      parseFloat(y),
      parseFloat(width),
      parseFloat(height),
    );
    if (inViewportRatio < 0.5) {
      count += 1;
      removeNodeInGraph(node, nodeidToCursor, accessibilityTree);
    }
  });
  console.log('number of nodes marked:', count);
  accessibilityTree = accessibilityTree.filter(node => node.parentId !== "[REMOVED]");
  return accessibilityTree;
}

function getInViewportRatio(elemLeftBound, elemTopBound, width, height, config) {
  const elemRightBound = elemLeftBound + width;
  const elemLowerBound = elemTopBound + height;

  const winLeftBound = 0;
  // Ensure we access winWidth and winHeight from the config object correctly
  const winRightBound = 1024; // Adjusted to access winWidth
  const winTopBound = 0;
  const winLowerBound = 768; // Adjusted to access winHeight

  // Compute the overlap in x and y axes
  const overlapWidth = Math.max(
      0,
      Math.min(elemRightBound, winRightBound) - Math.max(elemLeftBound, winLeftBound),
  );
  const overlapHeight = Math.max(
      0,
      Math.min(elemLowerBound, winLowerBound) - Math.max(elemTopBound, winTopBound),
  );

  // Compute the overlap area
  const ratio = (overlapWidth * overlapHeight) / (width * height);
  return ratio;
}

app.post('/getAccessibilityTree', async (req, res) => {
  const { browserId, pageId, currentRound } = req.body;

  if (!browserId || !pageId) {
    return res.status(400).send({ error: 'Missing browserId or pageId.' });
  }

  const pageEntry = findPageByPageId(browserId, pageId); 
  const page = pageEntry.page;
  if (!page) {
    return res.status(404).send({ error: 'Page not found.' });
  }

  try {
    
    console.time('FullAXTTime');
    const client = await page.context().newCDPSession(page);
    const response = await client.send('Accessibility.getFullAXTree');
    //logMemoryUsage();
    const [axtree, backendDOMids] = await fetchPageAccessibilityTree(response.nodes);
    //logMemoryUsage();
    console.log('finished fetching page accessibility tree')
    const boundingClientRects = await fetchAllBoundingClientRects(client, backendDOMids);
    //logMemoryUsage();
    console.log('finished fetching bounding client rects')
    console.log('boundingClientRects:', boundingClientRects.length, 'axtree:', axtree.length);
    // assign boundingClientRects to axtree
    for (let i = 0; i < boundingClientRects.length; i++) {
      if (axtree[i].role.value === 'RootWebArea') {
        axtree[i].union_bound = [0.0, 0.0, 10.0, 10.0];
      } else {
        axtree[i].union_bound = boundingClientRects[i].result.value;
      }
      // if name and value are both defined
      // if (axtree[i].name && axtree[i].name.value && axtree[i].name.value == '2024') {
      //   console.log('boundingClientRect:', axtree[i]);
      // }
    }
    const pruned_axtree = processAccessibilityTree(axtree);
    // const locators = await Promise.all(
    //   pruned_axtree.map(async (node) => {
    //     return getLocatorFromNodeId(page, node);
    //     //return resolveNodeFromBackendNodeId(page.mainFrame(), node.backendDOMNodeId)
    //   })
    // );
    // const elementsWithAttributes = await page.evaluate(() => {
    //   const elements = document.querySelectorAll('*');
    //   const elementsWithAttributes = [];
    //   elements.forEach(element => {
    //     const attributes = {};
    //     for (const attr of element.attributes) {
    //       attributes[attr.name] = attr.value;
    //     }
    //     elementsWithAttributes.push({
    //       tagName: element.tagName,
    //       attributes: attributes
    //     });
    //   });
    //   return elementsWithAttributes;
    // });
    // console.log('number of elements:', elementsWithAttributes);
    // for (let i = 0; i < elementsWithAttributes.length; i++) {
    //   console.log(elementsWithAttributes[i]);
    // }

    // attach the element handles to the pruned axtree
    // pruned_axtree.forEach((node, idx) => {
    //   node.locator = locators[idx];
    // });
    // console.log('finished getting element handles', locators.length, pruned_axtree.length);
    const {treeStr, treeIdxtoElement} = parseAccessibilityTree(pruned_axtree);
    console.timeEnd('FullAXTTime');
    console.log(treeStr);
    pageEntry['treeIdxtoElement'] = treeIdxtoElement;
    const accessibilitySnapshot = await page.accessibility.snapshot();
    // console.log(JSON.stringify(accessibilitySnapshot, null, 2));

    const prefix = findPagePrefixesWithCurrentMark(browserId, pageId) || '';
    let yamlWithPrefix = `${prefix}\n\n${treeStr}`;

    if (pageEntry['downloadedFiles'].length > 0) {
      if (pageEntry['downloadSources'].length < pageEntry['downloadedFiles'].length) {
        const source_name = pruned_axtree[0].name.value;
        while (pageEntry['downloadSources'].length < pageEntry['downloadedFiles'].length) {
          pageEntry['downloadSources'].push(source_name);
        }
      }
      const downloadedFiles = pageEntry['downloadedFiles'];
      // add the downloaded files to the yaml
      yamlWithPrefix += `\n\nYou have successfully downloaded the following files:\n`;
      downloadedFiles.forEach((file, idx) => {
        yamlWithPrefix += `File ${idx + 1} (from ${pageEntry['downloadSources'][idx]}): ${file}\n`;
      }
      );
    }

    // await page.evaluate(() => {
    //   window.detectDOMChanges = () => {
    //     return new Promise((resolve) => {
    //       const observer = new MutationObserver(() => {
    //         observer.disconnect();
    //         resolve(true);
    //       });

    //       observer.observe(document.body, { childList: true, subtree: true });

    //       // Timeout after 2 seconds if no changes are detected
    //       setTimeout(() => {
    //         observer.disconnect();
    //         resolve(false);
    //       }, 2000);
    //     });
    //   };
    // });

    const screenshotBuffer = await page.screenshot();

    const fileName = `${browserId}@@${pageId}@@${currentRound}.png`;
    const filePath = path.join('/screenshots', fileName);

    await fs.writeFile(filePath, screenshotBuffer);
    await getboxedScreenshot(page, browserId, pageId, currentRound, treeIdxtoElement);
    const currentUrl = page.url();
    res.send({ yaml: yamlWithPrefix, url: currentUrl, snapshot: accessibilitySnapshot, treeIdxtoElement: treeIdxtoElement});
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to get accessibility tree.' });
  }
});

app.post('/getAccessibilityTreeOnly', async (req, res) => {
  const { browserId, pageId } = req.body;

  if (!browserId || !pageId) {
    return res.status(400).send({ error: 'Missing browserId or pageId.' });
  }

  const pageEntry = findPageByPageId(browserId, pageId); 
  const page = pageEntry.page;
  if (!page) {
    return res.status(404).send({ error: 'Page not found.' });
  }

  try {
    
    console.time('FullAXTTime');
    const client = await page.context().newCDPSession(page);
    const response = await client.send('Accessibility.getFullAXTree');
    //logMemoryUsage();
    const [axtree, backendDOMids] = await fetchPageAccessibilityTree(response.nodes);
    //logMemoryUsage();
    console.log('finished fetching page accessibility tree')
    const boundingClientRects = await fetchAllBoundingClientRects(client, backendDOMids);
    //logMemoryUsage();
    console.log('finished fetching bounding client rects')
    console.log('boundingClientRects:', boundingClientRects.length, 'axtree:', axtree.length);
    // assign boundingClientRects to axtree
    for (let i = 0; i < boundingClientRects.length; i++) {
      if (axtree[i].role.value === 'RootWebArea') {
        axtree[i].union_bound = [0.0, 0.0, 10.0, 10.0];
      } else {
        axtree[i].union_bound = boundingClientRects[i].result.value;
      }
      // if name and value are both defined
      // if (axtree[i].name && axtree[i].name.value && axtree[i].name.value == '2024') {
      //   console.log('boundingClientRect:', axtree[i]);
      // }
    }
    const pruned_axtree = processAccessibilityTree(axtree);
    // const locators = await Promise.all(
    //   pruned_axtree.map(async (node) => {
    //     return getLocatorFromNodeId(page, node);
    //     //return resolveNodeFromBackendNodeId(page.mainFrame(), node.backendDOMNodeId)
    //   })
    // );
    // const elementsWithAttributes = await page.evaluate(() => {
    //   const elements = document.querySelectorAll('*');
    //   const elementsWithAttributes = [];
    //   elements.forEach(element => {
    //     const attributes = {};
    //     for (const attr of element.attributes) {
    //       attributes[attr.name] = attr.value;
    //     }
    //     elementsWithAttributes.push({
    //       tagName: element.tagName,
    //       attributes: attributes
    //     });
    //   });
    //   return elementsWithAttributes;
    // });
    // console.log('number of elements:', elementsWithAttributes);
    // for (let i = 0; i < elementsWithAttributes.length; i++) {
    //   console.log(elementsWithAttributes[i]);
    // }

    // attach the element handles to the pruned axtree
    // pruned_axtree.forEach((node, idx) => {
    //   node.locator = locators[idx];
    // });
    // console.log('finished getting element handles', locators.length, pruned_axtree.length);
    const {treeStr, treeIdxtoElement} = parseAccessibilityTree(pruned_axtree);
    console.timeEnd('FullAXTTime');
    console.log(treeStr);
    pageEntry['treeIdxtoElement'] = treeIdxtoElement;
    const accessibilitySnapshot = await page.accessibility.snapshot();
    // console.log(JSON.stringify(accessibilitySnapshot, null, 2));

    const prefix = findPagePrefixesWithCurrentMark(browserId, pageId) || '';
    let yamlWithPrefix = `${prefix}\n\n${treeStr}`;

    if (pageEntry['downloadedFiles'].length > 0) {
      if (pageEntry['downloadSources'].length < pageEntry['downloadedFiles'].length) {
        const source_name = pruned_axtree[0].name.value;
        while (pageEntry['downloadSources'].length < pageEntry['downloadedFiles'].length) {
          pageEntry['downloadSources'].push(source_name);
        }
      }
      const downloadedFiles = pageEntry['downloadedFiles'];
      // add the downloaded files to the yaml
      yamlWithPrefix += `\n\nYou have successfully downloaded the following files:\n`;
      downloadedFiles.forEach((file, idx) => {
        yamlWithPrefix += `File ${idx + 1} (from ${pageEntry['downloadSources'][idx]}): ${file}\n`;
      }
      );
    }

    // await page.evaluate(() => {
    //   window.detectDOMChanges = () => {
    //     return new Promise((resolve) => {
    //       const observer = new MutationObserver(() => {
    //         observer.disconnect();
    //         resolve(true);
    //       });

    //       observer.observe(document.body, { childList: true, subtree: true });

    //       // Timeout after 2 seconds if no changes are detected
    //       setTimeout(() => {
    //         observer.disconnect();
    //         resolve(false);
    //       }, 2000);
    //     });
    //   };
    // });

    // const screenshotBuffer = await page.screenshot();

    // const fileName = `${browserId}@@${pageId}@@${currentRound}.png`;
    // const filePath = path.join('/screenshots', fileName);

    // await fs.writeFile(filePath, screenshotBuffer);
    // await getboxedScreenshot(page, browserId, pageId, currentRound, treeIdxtoElement);
    const currentUrl = page.url();
    res.send({ yaml: yamlWithPrefix, url: currentUrl, snapshot: accessibilitySnapshot, treeIdxtoElement: treeIdxtoElement});
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to get accessibility tree.' });
  }
});

async function getboxedScreenshot(page, browserId, pageId, currentRound, treeIdxtoElement) {
  // filter treeIdxtoElement to only include elements that are interactive 
  // (e.g., buttons, links, form elements, etc.)
  const interactiveElements = {};
  Object.keys(treeIdxtoElement).forEach(function(index) {
    var elementData = treeIdxtoElement[index];
    var role = elementData.role.value;
    if (role === 'button' || role === 'link' || role === 'tab' || role.includes('box')) {
      interactiveElements[index] = elementData;
    }
  });

  await page.evaluate((interactiveElements) => {
    Object.keys(interactiveElements).forEach(function(index) {
      var elementData = interactiveElements[index];
      var unionBound = elementData.union_bound; // Access the union_bound object
      
      // Create a new div element to represent the bounding box
      var newElement = document.createElement("div");
      var borderColor = '#000000'; // Use your color function to get the color
      newElement.style.outline = `2px dashed ${borderColor}`;
      newElement.style.position = "fixed";
      
      // Use union_bound's x, y, width, and height
      newElement.style.left = unionBound.x + "px";
      newElement.style.top = unionBound.y + "px";
      newElement.style.width = unionBound.width + "px";
      newElement.style.height = unionBound.height + "px";
      
      newElement.style.pointerEvents = "none";
      newElement.style.boxSizing = "border-box";
      newElement.style.zIndex = 2147483647;
      newElement.classList.add("bounding-box"); 
      
      // Create a floating label to show the index
      var label = document.createElement("span");
      label.textContent = index;
      label.style.position = "absolute";
      
      // Adjust label position with respect to union_bound
      label.style.top = Math.max(-19, -unionBound.y) + "px";
      label.style.left = Math.min(Math.floor(unionBound.width / 5), 2) + "px";
      label.style.background = borderColor;
      label.style.color = "white";
      label.style.padding = "2px 4px";
      label.style.fontSize = "12px";
      label.style.borderRadius = "2px";
      newElement.appendChild(label);
      
      // Append the element to the document body
      document.body.appendChild(newElement);
    });
  }, interactiveElements);  // Pass treeIdxtoElement here as a second argument

  // Optionally wait a bit to ensure the boxes are drawn
  await page.waitForTimeout(1000);

  // Take the screenshot
  const screenshotBuffer = await page.screenshot();

  // Define the file name and path
  const fileName = `${browserId}@@${pageId}@@${currentRound}_with_box.png`;
  const filePath = path.join('/screenshots', fileName);

  // Write the screenshot to a file
  await fs.writeFile(filePath, screenshotBuffer);

  await page.evaluate(() => {
    document.querySelectorAll(".bounding-box").forEach(box => box.remove());
  });
}


async function adjustAriaHiddenForSubmenu(menuitemElement) {
  try {
    // Find the submenu associated with the menuitem
    const submenu = await menuitemElement.$('div.submenu');
    if (submenu) {
      await submenu.evaluate(node => {
        node.setAttribute('aria-hidden', 'false');
      });
      
      // Optionally, if you need to adjust further attributes or elements within the submenu
      // const submenuItems = await submenu.$$('li.submenu_item');
      // for (const submenuItem of submenuItems) {
      //   await submenuItem.evaluate(node => {
      //     node.setAttribute('aria-hidden', 'false');
      //     node.removeAttribute('aria-label');
      //   });
      // }
    }
  } catch (e) {
    console.log('Failed to adjust aria-hidden for submenu:', e);
  }
}

async function clickElement(click_locator, adjust_aria_label, x1, x2, y1, y2) {
  //for (const element of await click_locator.all()) {
  const elements = adjust_aria_label ? await click_locator.elementHandles() : await click_locator.all();
  // print the number of elements
  console.log('number of elements for click:', elements.length);
  // print the elements themselves
  //console.log('elements:', elements);
  // Ensure the click will only change the current tab by setting the target attribute to '_self'
  if (elements.length > 1) {
    for (const element of elements) {
      await element.evaluate(el => {
        if (el.tagName.toLowerCase() === 'a' && el.hasAttribute('target')) {
          el.setAttribute('target', '_self');
        }
      });
    }
    const targetX = (x1 + x2) / 2;
    const targetY = (y1 + y2) / 2;

    let closestElement = null;
    let closestDistance = Infinity;

    for (const element of elements) {
      const boundingBox = await element.boundingBox();
      if (boundingBox) {
        const elementCenterX = boundingBox.x + boundingBox.width / 2;
        const elementCenterY = boundingBox.y + boundingBox.height / 2;

        const distance = Math.sqrt(
          Math.pow(elementCenterX - targetX, 2) + Math.pow(elementCenterY - targetY, 2)
        );
        if (distance < closestDistance) {
          closestDistance = distance;
          closestElement = element;
        }
      }
    }
    await closestElement.click({ timeout: 5000, force: true});
    if (adjust_aria_label) {
      await adjustAriaHiddenForSubmenu(closestElement);
    }
  } else if (elements.length === 1) {
    await elements[0].evaluate(el => {
      if (el.tagName.toLowerCase() === 'a' && el.hasAttribute('target')) {
        el.setAttribute('target', '_self');
      }
    });
    await elements[0].click({ timeout: 5000, force: true});
    if (adjust_aria_label) {
      await adjustAriaHiddenForSubmenu(elements[0]);
    }
  } else {
    return false;
  }
  return true;
  // let click_passed = 0;
  // for (const element of elements) {
  //   try {
  //     await element.click({ timeout: 5000, force: true });
  //     // const isChanged = await page.evaluate(async () => {
  //     //   return await window.detectDOMChanges();
  //     // });
  //     // if (click_only && !isChanged) {
  //     //   console.log('DOM did not change after click');
  //     //   continue;
  //     // }
  //     if (adjust_aria_label) {
  //       await adjustAriaHiddenForSubmenu(element);
  //     }
  //     click_passed += 1;
  //     //return true;  // Click successful
  //   } catch (e) {
  //     console.log('click failed:', e);
  //     continue;
  //   }
  // }
  // if (click_passed > 0){
  //   return true;  // Click successful
  // }
  // return false;
}

app.post('/performAction', async (req, res) => {
  const { browserId, pageId, actionName, targetId, targetElementType, targetElementName, actionValue, needEnter } = req.body;

  if (['click', 'type'].includes(actionName) && (!browserId || !actionName || !targetElementType || !pageId)) {
      return res.status(400).send({ error: 'Missing required fields.' });
  } else if (!browserId || !actionName || !pageId) {
      return res.status(400).send({ error: 'Missing required fields.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (!browserEntry || !browserEntry.browser) {
      return res.status(404).send({ error: 'Browser not found.' });
  }

  const pageEntry = browserEntry.pages[pageId];
  if (!pageEntry || !pageEntry.page) {
      return res.status(404).send({ error: 'Page not found.' });
  }
  try {
      const page = pageEntry.page;
      const treeIdxtoElement = pageEntry.treeIdxtoElement;
      //let selector = `${targetElementType}[name="${targetElementName}"]`;
      let adjust_aria_label = false;
      // if targetelementType is menuitem, then set the flag to true
      if (targetElementType === 'menuitem' || targetElementType === 'combobox') {
        adjust_aria_label = true;
      }
      //console.log(`selector: ${selector}`)
      //console.log('treeIdxtoElement:', treeIdxtoElement.length);
      //console.log('targetId:', targetId);
      //console.log(treeIdxtoElement[Number(targetId)]);
      //console.log(treeIdxtoElement[targetId]);
      switch (actionName) {
          case 'click':
            let element = treeIdxtoElement[targetId];
            //let clicked = false;
            //await element.click({ timeout: 5000, force: true });
            // await page.mouse.click(element.union_bound.x + element.union_bound.width / 2, element.union_bound.y + element.union_bound.height / 2);
            // if (adjust_aria_label) {
            //   await adjustAriaHiddenForSubmenu(element);
            // }
            if (targetElementType === 'Iframe'){
              const iframes = await page.frames();
              console.log(`Found ${iframes.length} iframes`);

              for (const frame of iframes) {
                console.log(`Checking iframe: ${frame.url()}`);
                try {
                  // Find the button with the text "Sign in with Google Button" inside the iframe
                  const button = await frame.frameLocator(`title="${targetElementName}"`).first();
                  if (button) {
                    // Click the button if found
                    await button.click({ timeout: 5000, force: true });
                    console.log('IFrame Button clicked');
                    break; // Exit the loop once the button is clicked
                  }
                } catch (e) {
                  // Ignore errors and continue checking the next iframe
                }
              }
            } else {
              let clicked = false;
              let click_locator;
              try{
                click_locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000});
                clicked = await clickElement(click_locator, adjust_aria_label, element.union_bound.x, element.union_bound.x + element.union_bound.width, element.union_bound.y, element.union_bound.y + element.union_bound.height);
              } catch (e) {
                console.log(e);
                clicked = false;
              }
              if (!clicked) {
                //let node = treeIdxtoElement[targetId];
                //await page.mouse.click(node.union_bound.x + node.union_bound.width / 2, node.union_bound.y + node.union_bound.height / 2);
                const click_locator = await page.getByRole(targetElementType, { name: targetElementName});
                clicked = await clickElement(click_locator, adjust_aria_label, element.union_bound.x, element.union_bound.x + element.union_bound.width, element.union_bound.y, element.union_bound.y + element.union_bound.height);
                if (!clicked) {
                  // get only the first three words form the targetElementName and try partial match
                  const targetElementNameStartWords = targetElementName.split(' ').slice(0, 3).join(' ');
                  const click_locator = await page.getByText(targetElementNameStartWords);
                  clicked = await clickElement(click_locator, adjust_aria_label, element.union_bound.x, element.union_bound.x + element.union_bound.width, element.union_bound.y, element.union_bound.y + element.union_bound.height);
                  if (!clicked) {
                    return res.status(400).send({ error: 'No clickable element found.' });
                  }
                }
              }
            }
            await page.waitForTimeout(5000); 
            break;
          case 'type':
              let type_clicked = false;
              let locator;
              let node = treeIdxtoElement[targetId];
              try{
                locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000}).first() 
                type_clicked = await clickElement(locator, adjust_aria_label, node.union_bound.x, node.union_bound.x + node.union_bound.width, node.union_bound.y, node.union_bound.y + node.union_bound.height);
              } catch (e) {
                console.log(e);
                type_clicked = false;
              }
              // if (!type_clicked) {
                
              //   await page.mouse.click(node.union_bound.x + node.union_bound.width / 2, node.union_bound.y + node.union_bound.height / 2);
              // }
              if (!type_clicked) {
                locator = await page.getByRole(targetElementType, { name: targetElementName}).first() 
                type_clicked = await clickElement(locator, adjust_aria_label, node.union_bound.x, node.union_bound.x + node.union_bound.width, node.union_bound.y, node.union_bound.y + node.union_bound.height);
                console.log('after non-exact type_clicked:', type_clicked);
                if (!type_clicked) {
                  locator = await page.getByPlaceholder(targetElementName).first();
                  type_clicked = await clickElement(locator, adjust_aria_label, node.union_bound.x, node.union_bound.x + node.union_bound.width, node.union_bound.y, node.union_bound.y + node.union_bound.height);
                  console.log('after get by placeholder type_clicked:', type_clicked);
                  if (!type_clicked) {
                    return res.status(400).send({ error: 'No clickable element found.' });
                  }
                }
              }
              
              // const isClickable = await locator.evaluate(node => {
              //   const rect = node.getBoundingClientRect();
              //   const elementFromPoint = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
              //   return elementFromPoint === node;
              // });
              //if (!isClickable) {
                //console.log('Element is not clickable.');
              // locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000});
              // console.log('new locator:', locator);
              // const elements = await page.$$('role=combobox');
  
              // // Iterate through the elements to find the one with the specified name
              // for (const element of elements) {
              //   const name = await element.evaluate(node => node.innerText);
              //   const backendid = await element.evaluate(node => node.backendDOMNodeId);
              //   console.log('found the element:', backendid, name);
              // }
                          //}
              
              // // using XPath to find the element, e.g. targetElementName, targetElementType, parentId
              // const xpath = `//*[@role="${targetElementType}" and @name="${targetElementName}" and @parentId="${node.parentId}" and @nodeId="${node.nodeId}"]`;
              // const locator = await page.locator(xpath);
              
              // console.log(locator);
              // await locator.click();
              
              await page.keyboard.press('Control+A');
              await page.keyboard.press('Backspace');
              // await locator.asElement().click({ timeout: 5000, force: true });
              // await locator.press('Control+A'); 
              // await locator.press('Backspace');            
              if (needEnter) {
                const newactionValue = actionValue + '\n';
                await page.keyboard.type(newactionValue);
              } else {
                await page.keyboard.type(actionValue);
              }
              
              break;
          case 'select':
              let menu = treeIdxtoElement[targetId];
              let menu_locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000});
              console.log('Selecting an option:', actionValue);
              await menu_locator.selectOption({ label: actionValue })
              await menu_locator.click();
              break;
          case 'scroll':
              if (actionValue === 'down') {
                  await page.evaluate(() => window.scrollBy(0, window.innerHeight));
              } else if (actionValue === 'up') {
                  await page.evaluate(() => window.scrollBy(0, -window.innerHeight));
              } else {
                  return res.status(400).send({ error: 'Unsupported scroll direction.' });
              }
              break;
          case 'goback':
              await page.goBack();
              break;
          case 'restart':
              await page.goto(actionValue !== null && actionValue !== undefined ? actionValue : "https://www.google.com");
              break;
          case 'wait':
              // sleep for 3 seconds
              await sleep(3000);
              break;
          default:
              return res.status(400).send({ error: 'Unsupported action.' });
      }

      browserEntry.lastActivity = Date.now();
      await sleep(3000); 
      const currentUrl = page.url();
      console.log(`current url: ${currentUrl}`);
      res.send({ message: 'Action performed successfully.' , currentUrl: currentUrl});
  } catch (error) {
      console.error(error);
      res.status(500).send({ error: 'Failed to perform action.' });
  }
});

// app.post('/performAction', async (req, res) => {
//   const { browserId, pageId, actionName, targetElementType, targetElementName, actionValue, needEnter } = req.body;

//   if (['click', 'type'].includes(actionName) && (!browserId || !actionName || !targetElementType || !pageId)) {
//       return res.status(400).send({ error: 'Missing required fields.' });
//   } else if (!browserId || !actionName || !pageId) {
//       return res.status(400).send({ error: 'Missing required fields.' });
//   }

//   const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
//   const browserEntry = browserPool[slot]
//   if (!browserEntry || !browserEntry.browser) {
//       return res.status(404).send({ error: 'Browser not found.' });
//   }

//   const pageEntry = browserEntry.pages[pageId];
//   if (!pageEntry || !pageEntry.page) {
//       return res.status(404).send({ error: 'Page not found.' });
//   }

//   try {
//       const page = pageEntry.page;

//       let selector = `${targetElementType}[name="${targetElementName}"]`;
//       let adjust_aria_label = false;
//       // if targetelementType is menuitem, then set the flag to true
//       if (targetElementType === 'menuitem') {
//         adjust_aria_label = true;
//       }
//       console.log(`selector: ${selector}`)
//       switch (actionName) {
//           case 'click':
//             let clicked = false;
//             let click_locator;
//             try{
//               click_locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000});
//               clicked = await clickElement(click_locator, adjust_aria_label, page, true);
//             } catch (e) {
//               console.log(e);
//               clicked = false;
//             }
//             if (!clicked) {
//               const click_locator = await page.getByRole(targetElementType, { name: targetElementName});
//               clicked = await clickElement(click_locator, adjust_aria_label, page, true);
//               if (!clicked) {
//                 // get only the first three words form the targetElementName and try partial match
//                 const targetElementNameStartWords = targetElementName.split(' ').slice(0, 3).join(' ');
//                 const click_locator = await page.getByText(targetElementNameStartWords);
//                 clicked = await clickElement(click_locator, adjust_aria_label, page, true);
//                 if (!clicked) {
//                   return res.status(400).send({ error: 'No clickable element found.' });
//                 }
//               }
//             }
//             break;
//           case 'type':
//               let type_clicked = false;
//               let locator;
//               try{
//                 locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000}).first() 
//                 type_clicked = await clickElement(locator, adjust_aria_label, page);
//               } catch (e) {
//                 console.log(e);
//                 type_clicked = false;
//               }
//               if (!type_clicked) {
//                 locator = await page.getByRole(targetElementType, { name: targetElementName}).first() 
//                 type_clicked = await clickElement(locator, adjust_aria_label, page);
//                 console.log('after non-exact type_clicked:', type_clicked);
//                 if (!type_clicked) {
//                   locator = await page.getByPlaceholder(targetElementName).first();
//                   type_clicked = await clickElement(locator, adjust_aria_label, page);
//                   console.log('after get by placeholder type_clicked:', type_clicked);
//                   if (!type_clicked) {
//                     return res.status(400).send({ error: 'No clickable element found.' });
//                   }
//                 }
//               }
//               await locator.press('Control+A'); 
//               await locator.press('Backspace');            
//               if (needEnter) {
//                 const newactionValue = actionValue + '\n';
//                 await page.keyboard.type(newactionValue);
//               } else {
//                 await page.keyboard.type(actionValue);
//               }
              
//               break;
//           case 'scroll':
//               if (actionValue === 'down') {
//                   await page.evaluate(() => window.scrollBy(0, window.innerHeight));
//               } else if (actionValue === 'up') {
//                   await page.evaluate(() => window.scrollBy(0, -window.innerHeight));
//               } else {
//                   return res.status(400).send({ error: 'Unsupported scroll direction.' });
//               }
//               break;
//           case 'goback':
//               await page.goBack();
//               break;
//           case 'restart':
//               await page.goto("https://www.google.com");
//               break;
//           case 'wait':
//               // sleep for 3 seconds
//               await sleep(3000);
//               break;
//           default:
//               return res.status(400).send({ error: 'Unsupported action.' });
//       }

//       browserEntry.lastActivity = Date.now();
//       await sleep(3000); 
//       const currentUrl = page.url();
//       console.log(`current url: ${currentUrl}`);
//       res.send({ message: 'Action performed successfully.' });
//   } catch (error) {
//       console.error(error);
//       res.status(500).send({ error: 'Failed to perform action.' });
//   }
// });

app.post('/takeScreenshot', async (req, res) => {
  const { browserId, pageId } = req.body;

  if (!browserId || !pageId) {
    return res.status(400).send({ error: 'Missing required fields: browserId, pageId.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (!browserEntry || !browserEntry.browser) {
    return res.status(404).send({ error: 'Browser not found.' });
  }

  const pageEntry = browserEntry.pages[pageId];
  if (!pageEntry || !pageEntry.page) {
    return res.status(404).send({ error: 'Page not found.' });
  }

  try {
    const page = pageEntry.page;
    const screenshotBuffer = await page.screenshot({ fullPage: true });

    res.setHeader('Content-Type', 'image/png');
    res.send(screenshotBuffer);
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to take screenshot.' });
  }
});

app.post('/loadScreenshot', (req, res) => {
  const { browserId, pageId, currentRound } = req.body;
  const fileName = `${browserId}@@${pageId}@@${currentRound}.png`;
  const filePath = path.join('/screenshots', fileName);

  res.sendFile(filePath, (err) => {
    if (err) {
      console.error(err);
      if (err.code === 'ENOENT') {
        res.status(404).send({ error: 'Screenshot not found.' });
      } else {
        res.status(500).send({ error: 'Error sending screenshot file.' });
      }
    }
  });
});

app.post('/gethtmlcontent', async (req, res) => {
  const { browserId, pageId, currentRound } = req.body;
  // if (!browserId || !pageId) {
  //   return res.status(400).send({ error: 'Missing browserId or pageId.' });
  // }
  const pageEntry = findPageByPageId(browserId, pageId); 
  const page = pageEntry.page;
  // if (!page) {
  //   return res.status(404).send({ error: 'Page not found.' });
  // }
  try {
    const html = await page.content();
    const currentUrl = page.url();
    res.send({ html: html, url: currentUrl });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to get html content.' });
  }
});

app.listen(port, () => {
  initializeBrowserPool(maxBrowsers);
  console.log(`Server listening at http://localhost:${port}`);
});


process.on('exit', async () => {
  for (const browserEntry of browserPool) {
      await browserEntry.browser.close();
  }
});
