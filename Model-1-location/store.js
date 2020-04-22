
// read in requirements
const puppeteer = require('puppeteer');
const {Storage} = require('@google-cloud/storage');

// defining cloud storage id and bucket name
const GOOGLE_CLOUD_PROJECT_ID = "airqo-250220";
const BUCKET_NAME = "traffic_maps";

// The function that initiates saving to the cloud
exports.run = async (req, res) => {
  res.setHeader("content-type", "application/json");
  console.log('underway')
  // do we have a buffer file and a screenshot destination url
  try {
    const buffer = await screenshotmaps(req.body);
    let screenshotUrl = await uploadToGoogleCloud(buffer, pairs.loc_ref + '_' + datestamp);
    console.log(screenshot.url)
    res.status(200).send(JSON.stringify({
      'screenshotUrl': screenshotUrl
    }));
    
  } catch(error) {
    res.status(422).send(JSON.stringify({
      error: error.message,
    }));
  }
};

// Takes the buffer and the filename and saves to cloud as a url
async function uploadToGoogleCloud(buffer, filename) {
    const storage = new Storage({
        projectId: GOOGLE_CLOUD_PROJECT_ID,
    });

    const bucket = storage.bucket(BUCKET_NAME);

    const file = bucket.file(filename);
    await uploadBuffer(file, buffer, filename);
  
    await file.makePublic();

  	return `https://${BUCKET_NAME}.storage.googleapis.com/${filename}`;
}
// takes the screenshot and returns a buffer
// https://www.bram.us/2019/12/02/building-a-website-screenshot-api-with-puppeteer-and-google-cloud-functions/
// async function takeScreenshot(params) {
// 	const browser = await puppeteer.launch({
// 		args: ['--no-sandbox']
// 	});
// 	const page = await browser.newPage();
// 	await page.goto(params.url, {waitUntil: 'networkidle2'});

// 	const buffer = await page.screenshot();

// 	await page.close();
// 	await browser.close();
  
//   	return buffer;
// }


/// takes screenshot and returns buffer pg
async function screenshotmaps () {  
    // load repository details for this array of repo URLs
    const urls = [
   {'loc_ref': 'loc_23',
    'display': 'MAGWA WARD, JINJA',
    'chan_id': 689753,
    'url': 'https://www.google.com/maps/@0.432968,33.20001,15z/data=!5m1!1e1'},
    {'loc_ref': 'loc_26',
    'display': 'CIVIC CENTRE, KAMPALA',
    'chan_id': 689761,
    'url': 'https://www.google.com/maps/@0.314,32.580000000000005,15z/data=!5m1!1e1'}
   ];
  
    var today = new Date();
    var datestamp = "_" + today.getFullYear() + "-" + (today.getMonth()+1) + "-" + today.getDate() + "_" + today.getHours() + '-' + today.getMinutes() + '.png';
  
      urls.map(async run => {
          // looping through the locations and urls
          for (const [idx, pairs] of urls.entries()) {
              let browser = await puppeteer.launch({
                  args: ['--no-sandbox']
              });
              let page = await browser.newPage();
              await page.goto(pairs.url);
              console.log(pairs.url)
              await page.setViewport({ width: 1920, height: 1080 });
              await page.waitForNavigation(['load']);
              const buffer  = await page.screenshot({ 
              clip: { x: 550, y: 50, width: 1375, height: 889 } 
              });
              await page.close();
              await browser.close();
              
              return buffer
          };
      });
  };

// uploading buffer to cloud storage
async function uploadBuffer(file, buffer, filename) {
    return new Promise((resolve) => {
        file.save(buffer, { destination: filename }, () => {
            resolve();
        });
    })
}
exports.run()