

mids = [u'一般', u'还行', u'还算', u'凑合', u'平平']
goods = [u'好', u'干净', u'大', u'安静', u'不错', u'丰富', u'适中', u'佳', u'方便', u'便利', u'和蔼', u'舒适', u'热情', u'高', u'实惠', u'满意', u'齐全', u'近', u'卫生', u'优雅', '丰盛', u'整洁', u'优惠', u'到位', u'棒', u'便宜', u'幽雅', u'新', u'贴心', u'很棒', u'很大', u'优越', u'全', u'宽敞'， u'周到'， u'极佳', u'非常棒', u'完备', u'合适', u'较合适', u'规范', u'发达', u'合理', u'完善', u'赞', u'很丰盛',  u'一流', u'超高', u'喜欢', u'推荐', u'人性化', u'周到', u'可口']
bads = [u'旧', u'吵', u'陈旧', u'老旧', u'简陋', u'差', u'差劲', u'小', u'偏僻', u'冷淡', u'不好', u'不太好', u'远', u'偏', u'较差', u'漏风', u'差点', u'太差', u'敷衍', u'落伍', u'傲慢', u'太低', u'落后', u'老', u'少', u'改进', u'改善', u'简单', u'贵', u'马虎', u'不行', u'不行了', u'再', u'讨厌', u'偏远', u'没有', u'凶']

def infer(words, i):
  bu = False
  start = i + 1 
  if i != 0 and words[i - 1][0] == u'的':
    start = max(i - 2, 0)
  for j in range(i + 1, min(len(words), i + 5)):
    if j == i:
      continue
    if words[j][1] == 'n' and j - i > 2:
      break
    if words[j][0] == u'不' or words[j][1].startswith(u'不是'):
      print 'bu'
      bu = True
    elif words[j][0] in mids:
      return 1
    elif words[j][0] in goods:
      if words[i][0] == u'价格' and words[j][0] == u'高':
        return 2 if bu else 0
      return 0 if bu else 2
    elif words[j][0] in bads:
      return 2 if bu else 0
  return 0 if bu else 1

def tag(record):
  print record['text']
  for i, word in enumerate(record['words']):
    if word[0] in aspects:
      word[2] = infer(record['words'], i)
      print word[0], word[2]
    else:
      word[2] = -1
  # op = raw_input()
  #if op.strip():
  #   for i, word in enumerate(record['words']):
  #     if word[0] in aspects:
  #       word[2] = infer(record['words'], i)
  #       print word[0], word[2]
  #       op = raw_input()
  #       if op.strip():
  #         word[2] = int(op.strip())
  docs.save(record)

records = [record for record in docs.find()]
print len(records)
for k,record in enumerate(records):
  if isinstance(record['words'][0][2], int):
    continue
  print k
  tag(record)

