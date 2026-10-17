const {chromium} = require('playwright');
const path = require('path');
const {pathToFileURL} = require('url');
(async()=>{
 const root=path.resolve(__dirname,'..');
 const preview=process.argv.includes('--preview');
 const options={headless:true,args:['--allow-file-access-from-files']};
 if(process.env.PDF_BROWSER_PATH) options.executablePath=process.env.PDF_BROWSER_PATH;
 const browser=await chromium.launch(options);
 require('fs').mkdirSync(path.join(root,'output/pdf'),{recursive:true});
 const page=await browser.newPage();
 await page.goto(pathToFileURL(path.join(root,'tmp/pdfs/current_manuscript_print.html')).href);
 await page.evaluate(()=>document.fonts.ready);
 const bad=await page.evaluate(()=>Array.from(document.images).filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src));
 if(bad.length)throw Error('Missing images: '+bad);
 if(preview) await page.evaluate(async()=>{
  for(const img of document.images){
   await img.decode();
   const canvas=document.createElement('canvas');
   canvas.width=Math.min(img.naturalWidth,800);
   canvas.height=Math.round(canvas.width*img.naturalHeight/img.naturalWidth);
   canvas.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height);
   img.src=canvas.toDataURL('image/jpeg',0.82);
   await img.decode();
  }
 });
 const filename=preview?'molecular_space_time_low_resolution_preview.pdf':'molecular_space_time_latest_high_quality.pdf';
 await page.pdf({path:path.join(root,'output/pdf',filename),format:'A4',printBackground:true,preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span></div>'});
 await browser.close();
 console.log(filename+' written; all images loaded.');
})();
