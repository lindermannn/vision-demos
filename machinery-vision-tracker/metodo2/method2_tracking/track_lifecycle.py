def state(hits,misses,confirm_hits,max_lost_s,age_s):
 if hits<confirm_hits:return 'tentative'
 if misses==0:return 'confirmed'
 return 'removed' if age_s>max_lost_s else 'lost'
