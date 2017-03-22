import sys
import json
import time
import codecs
import random
import logging
import pymongo
import selenium
from selenium import webdriver

firefox = webdriver.Firefox()


def init_logger():
    name = "pageCrawler"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', '%a, %d %b %Y %H:%M:%S')
    file_handler = logging.FileHandler(name + ".log")
    file_handler.setFormatter(formatter)
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(std_handler)
    return logger


def read_page(hotelid, data, failed_id, success_id, ffl=False):
    if not ffl and success_id.find({"hotelid": hotelid}).count() != 0:
        return
    url = "http://m.ctrip.com/html5/hotel/HotelDetail/dianping/{}.html".format(hotelid)
    firefox.get(url)
    fl = False
    good_num = firefox.find_element_by_css_selector("span.js_tag_item:nth-child(2)").text
    good_num = int(good_num[good_num.find("(") + 1: good_num.find(")")])
    bad_num = firefox.find_element_by_css_selector("span.dp-derogatory:nth-child(3)").text
    bad_num = int(bad_num[bad_num.find("(") + 1: bad_num.find(")")])
    num = good_num + bad_num
    for i in range(10):
        try:
            firefox.find_elements_by_css_selector(".key-current")[0].click()
            time.sleep(0.5)
            fl = True
            break
        except Exception:
            pass
    if not fl:
        if ffl:
            failed_id.write("{}\n".format(hotelid))
        else:
            failed_id.save({"hotelid": hotelid})
        return False
    try:
        time.sleep(random.random() * 60)
        for _ in range(300):
            try:
                firefox.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.1)
                blocks = firefox.find_elements_by_css_selector(".hotel-t-border")
                if len(blocks) >= num:
                    break
                if ffl:
                    success_id.write("{}\n".format(hotelid))
                else:
                    success_id.save({"hotelid": hotelid})
            except selenium.common.exceptions.StaleElementReferenceException as e:
                pass
            except selenium.common.exceptions.NoSuchElementException as e:
                pass
    except selenium.common.exceptions.ElementNotVisibleException:
        logger.info("{} has reached its bottom!".format(url))
    except Exception, e:
        if ffl:
            failed_id.write("{}\n".format(hotelid))
        else:
            failed_id.save({"hotelid": hotelid})
        logger.error("Error: {}{}".format(hotelid, e))

    cnt = 0
    blocks = firefox.find_elements_by_css_selector(".hotel-t-border")
    for block in blocks:
        rk_elements = block.find_elements_by_css_selector(".g-ve")
        if len(rk_elements) == 1:
            cnt += 1
            rk = rk_elements[0].find_elements_by_tag_name("strong")[0].text
            text = block.find_elements_by_css_selector(".tree-ellips-line6")[0].text
            doc = {"rank": rk, "text": text, "hotelid": hotelid}
            if ffl:
                data.write(json.dumps(doc) + ",\n")
            else:
                data.save(doc)
    logger.info("Hotel {} has successfully extracted and saved {} records!".format(hotelid, cnt))
    return True


if __name__ == '__main__':
    logger = init_logger()
    ffl = False
    if not ffl:
        db = pymongo.MongoClient("localhost", 27017).paper
        data = db.tmpdata
        failed_id = db.failedid
        success_id = db.successid
    else:
        data = codecs.open("result.json", "w", "utf8")
        data.write("[")
        failed_id = open("failed_ids.txt", "w")
        success_id = open("failed_ids.txt", "w")

    fin = open("hotels.xiecheng.json")
    hotelMes = json.load(fin)
    ids = [mes["hotelId"] for mes in hotelMes]
    del hotelMes
    success_num = 0
    failed_num = 0
    for hotelid in ids:
        if read_page(hotelid.strip(), data, failed_id, success_id, ffl):
            success_num += 1
        else:
            failed_num += 1
    if ffl:
        data.write("]")
        data.close()
        failed_id.close()
        success_id.close()
    logger.info("{} cases success!".format(success_num))
    logger.info("{} cases failed!".format(failed_num))
