// https://www.scrapehero.com/how-to-take-screenshots-of-a-web-page-using-puppeteer/

var items = {'Kampala':'https://www.google.com/maps/dir/Entebbe/Kajjansi/@0.3206818,32.5481476,13.04z/data=!4m14!4m13!1m5!1m1!1s0x177d86b753c20ab3:0xa4a550c375cc2c14!2m2!1d32.463708!2d0.0511839!1m5!1m1!1s0x177d99d8aceb164f:0xb3b42975dfc7a0fa!2m2!1d32.5401025!2d0.2070254!3e0!5m1!1e1',
'Entebbe_Kamapla':'http://www.google.com/maps/dir/Kololo,+Kampala/Makerere+University,+University+Rd,+Kampala/@0.1472201,32.4932571,12z/data=!4m18!4m17!1m5!1m1!1s0x177dbba2a2d9eead:0x698c319a2299b891!2m2!1d32.5949199!2d0.327297!1m5!1m1!1s0x177dbb6d88e54def:0xddc6fcfbe10b089d!2m2!1d32.5710773!2d0.3292819!2m3!6e0!7e2!8j1585202400!3e0!5m1!1e1',
'Jinja_Kampala': 'https://www.google.com/maps/dir/Kololo,+Kampala/Makerere+University,+University+Rd,+Kampala/@0.3465628,32.8439372,11.48z/data=!4m18!4m17!1m5!1m1!1s0x177dbba2a2d9eead:0x698c319a2299b891!2m2!1d32.5949199!2d0.327297!1m5!1m1!1s0x177dbb6d88e54def:0xddc6fcfbe10b089d!2m2!1d32.5710773!2d0.3292819!2m3!6e0!7e2!8j1585202400!3e0!5m1!1e1'
};

const puppeteer = require('puppeteer');

async function run() {
    
    let browser = await puppeteer.launch({headless:false});
    let page = await browser.newPage();
    await page.goto(items[index]);
    await page.setViewport({ width: 1920, height: 1080 });
    await page.waitForNavigation(['load'])
    console.log(index)
    await page.screenshot({ path: './'+ index +'.png', type: 'png' , 
    clip: { x: 550, y: 50, width: 1375, height: 889 } 
    });
    await page.close();
    browser.close();
}

for(var index in items) {
    run();
    // browser.close();
}
// browser.close();
