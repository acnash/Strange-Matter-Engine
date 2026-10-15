const {chromium} = require('playwright');
const path = require('path');
const {pathToFileURL} = require('url');
(async()=>{
 const root=path.resolve(__dirname,'..');
 const options={headless:true};
 if(process.env.PDF_BROWSER_PATH) options.executablePath=process.env.PDF_BROWSER_PATH;
 const browser=await chromium.launch(options);
 require('fs').mkdirSync(path.join(root,'output/pdf'),{recursive:true});
 const page=await browser.newPage();
 await page.goto(pathToFileURL(path.join(root,'tmp/pdfs/current_manuscript_print.html')).href);
 await page.evaluate(()=>document.fonts.ready);
 const bad=await page.evaluate(()=>Array.from(document.images).filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src));
 if(bad.length)throw Error('Missing images: '+bad);
 await page.pdf({path:path.join(root,'output/pdf/molecular_space_time_latest_high_quality.pdf'),format:'A4',printBackground:true,preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span></div>'});
 await browser.close();
 console.log('High-quality PDF written; all images loaded.');
})();
